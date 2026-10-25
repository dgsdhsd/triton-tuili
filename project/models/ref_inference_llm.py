#!/usr/bin/env python3
"""
官方函数实现的推理 LLM (纯 PyTorch, 无自写 Triton kernel)

设计目标:
    作为 MOE-LLM.py 的"官方参照实现", 用于单序列(单组)推理场景。

对齐说明 (与 MOE-LLM.py 完全一致):
    - 结构: RMSNorm(Pre-LN) + Attention(官方 SDPA) + MoE(官方 matmul), 双残差
    - 权重: 直接加载 project/parameter/model_weights.pt (strict=True, W 完全对齐)
    - GQA:  num_heads=8, num_kv_heads=2, head_dim=64
    - MoE:  num_experts=8, top_k=2, d_ffn=1024
    - 无 RoPE, 无 dropout, 纯推理

"单组 vs 多组":
    单组 = 单条序列(batch=1)推理, 官方 SDPA 一次调用即可, 无需 cu_seqlens 变长划分。
    多组 = MOE-LLM.py 自写 kernel 支持的多变长序列 ragged 模式。

attention 用官方函数:
    torch.nn.functional.scaled_dot_product_attention (后端 cuDNN FlashAttention)
MoE 专家计算用官方函数:
    torch.matmul (A @ B^T, 与 grouped gemm 数学等价) + F.silu
"""

import importlib.util
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. 复用 MOE-LLM.py 的 GroupTransformer (仅用于主流程里做对照验证)
# =====================================================================
def load_moe_module():
    path = "/home/xkem/cuda_triton/project/MOE-LLM.py"
    spec = importlib.util.spec_from_file_location("moe_llm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# 2. 官方 MoE 层 (纯 PyTorch)
# =====================================================================
class RefMoE(nn.Module):
    """与 MoEInferenceLayer 结构/权重 key 完全对齐, 但专家计算用官方 matmul。"""

    def __init__(self, d_model, num_experts, top_k, d_ffn):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ffn = d_ffn

        # Router 路由权重 (与 MOE-LLM.py 一致, 全链路 fp16)
        self.router = nn.Linear(d_model, num_experts, bias=False, dtype=torch.float16)

        # 专家权重 (与 state_dict 的 moe_layer.w1 / moe_layer.w2 对齐)
        # w1: [num_experts, d_ffn, d_model]  -> up-proj  A[M, d_model] @ w1[e]^T -> [M, d_ffn]
        # w2: [num_experts, d_model, d_ffn]  -> down-proj A[M, d_ffn]  @ w2[e]^T -> [M, d_model]
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ffn, d_model, dtype=torch.float16))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ffn, dtype=torch.float16))

    def forward(self, x):
        # x: [N, d_model]
        N, D = x.shape

        # ---- 1. Router 打分 + Top-K 选择 (与 MOE-LLM.py 完全一致) ----
        router_logits = self.router(x)                       # [N, E]
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # ---- 2. Permute: 按专家排序 ----
        flat_indices = topk_indices.view(-1)                 # [N*top_k]
        flat_x = x.repeat_interleave(self.top_k, dim=0)      # [N*top_k, D]
        sorted_ids, permute = torch.sort(flat_indices)
        permuted_x = flat_x[permute]                         # [N*top_k, D]

        # ---- 3. 官方 matmul 逐专家计算 (无 CPU-GPU 同步, 对齐新版 MoE) ----
        # 旧版用 counts.tolist() + torch.split, 会强制 GPU→CPU 同步;
        # 这里改为纯 GPU 布尔掩码索引: 空专家得到 [0, D] 空矩阵,
        # 空矩阵 matmul 天然合法, torch.cat 自动忽略空切片, 全程无 .item()/.tolist()。
        outs = []
        for e in range(self.num_experts):
            mask = (sorted_ids == e)               # GPU 比较, 无同步
            xe = permuted_x[mask]                  # [count_e, D] (空专家为 [0, D])
            up = F.silu(xe @ self.w1[e].T)         # [count_e, d_ffn]  (up-proj + SiLU)
            down = up @ self.w2[e].T               # [count_e, D]      (down-proj)
            outs.append(down)
        permuted_out = torch.cat(outs, dim=0)      # [N*top_k, D], 无同步

        # ---- 4. Unpermute: 还原顺序 + 加权求和 ----
        unpermute = torch.empty_like(permute)
        unpermute[permute] = torch.arange(permute.size(0), device=x.device)
        restored = permuted_out[unpermute].view(N, self.top_k, D)
        return (restored * topk_weights.unsqueeze(-1)).sum(dim=1)   # [N, d_model]



# =====================================================================
# 3. 官方推理 LLM (纯 PyTorch, 单序列/单组)
# =====================================================================
class RefInferenceLLM(nn.Module):
    """与 GroupTransformer 结构/权重 key 完全对齐, attention 用官方 SDPA。"""

    def __init__(self, d_model=512, num_heads=8, num_kv_heads=2, head_dim=64,
                 num_experts=8, top_k=2, d_ffn=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.num_experts = num_experts

        # 归一化 (Pre-LN)
        self.input_norm = nn.RMSNorm(d_model, dtype=torch.float16)
        self.post_attn_norm = nn.RMSNorm(d_model, dtype=torch.float16)

        # Attention 投影
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=torch.float16)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=torch.float16)

        # MoE 专家层
        self.moe_layer = RefMoE(d_model, num_experts, top_k, d_ffn)

    def _attention(self, x_norm):
        # x_norm: [L, d_model]  单序列
        L, D = x_norm.shape
        H, KV, HD = self.num_heads, self.num_kv_heads, self.head_dim
        sm_scale = 1.0 / math.sqrt(HD)

        q = self.q_proj(x_norm).view(L, H, HD)
        k = self.k_proj(x_norm).view(L, KV, HD)
        v = self.v_proj(x_norm).view(L, KV, HD)

        # 转为官方 SDPA 需要的 [B, H, L, D]
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        # 官方 FlashAttention (GQA 通过 enable_gqa=True)
        o = F.scaled_dot_product_attention(
            q, k, v, enable_gqa=True, is_causal=True, scale=sm_scale
        )  # [1, H, L, D]

        o = o.transpose(1, 2).squeeze(0).reshape(L, H * HD)
        return self.out_proj(o)

    def forward(self, x):
        # x: [L, d_model]  单条序列
        residual = x
        x = residual + self._attention(self.input_norm(x))
        residual = x
        x = residual + self.moe_layer(self.post_attn_norm(x))
        return x


# =====================================================================
# 4. 主流程: 加载权重 -> 单序列推理 -> 与 MOE-LLM.py 的 GroupTransformer 对照
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    config_path = "/home/xkem/cuda_triton/project/parameter/model_config.json"
    weights_path = "/home/xkem/cuda_triton/project/parameter/model_weights.pt"

    # ---- 读取配置 ----
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ---- 构建官方参考模型并加载同一份权重 (W 对齐, strict=True) ----
    ref_model = RefInferenceLLM(**config).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    ref_model.load_state_dict(state_dict, strict=True)
    ref_model.eval()
    print(f"✅ 官方参考模型权重加载成功 (strict=True), 共 {len(state_dict)} 项")

    # ---- 构建 MOE-LLM.py 的 GroupTransformer (用于对照) ----
    moe = load_moe_module()
    mine_model = moe.GroupTransformer(**config).to(device)
    mine_model.load_state_dict(state_dict, strict=True)
    mine_model.eval()
    print("✅ MOE-LLM.py GroupTransformer 权重加载成功 (strict=True)\n")

    # ---- 单序列(单组)推理用例 ----
    seqlens = [60, 64, 100, 250, 512, 1000]

    def report(L, ref_out, mine_out):
        ref = ref_out.float()
        mine = mine_out.float()
        abs_diff = (ref - mine).abs()
        rel = abs_diff / ref.abs().clamp_min(1e-6)
        ok = torch.allclose(ref, mine, atol=1e-2, rtol=1e-2)
        print(
            f"[L={L:4d}] max_abs={abs_diff.max().item():.6e}  "
            f"mean_abs={abs_diff.mean().item():.6e}  "
            f"max_rel={rel.max().item():.6e}  mean_rel={rel.mean().item():.6e}  "
            f"allclose(1e-2)={ok}"
        )
        return ok

    results = []
    print("========== 单序列推理对照 (官方参考 vs 自写 Triton) ==========")
    for L in seqlens:
        x = torch.randn(L, config["d_model"], dtype=torch.float16, device=device)

        with torch.no_grad():
            ref_out = ref_model(x)                                        # 单序列, 官方函数
            cu = torch.tensor([0, L], dtype=torch.int32, device=device)   # 单序列的 cu_seqlens
            mine_out = mine_model(x, cu, L)                               # 喂给 GroupTransformer

        results.append(report(L, ref_out, mine_out))

    print("\n========== 汇总 ==========")
    print(f"通过 {sum(results)} / {len(results)} 用例")
    if all(results):
        print("✅ 官方参考推理 LLM 与自写 Triton 实现输出一致")
    else:
        print("❌ 存在不一致用例")

