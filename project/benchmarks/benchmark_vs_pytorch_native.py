#!/usr/bin/env python3
"""
方案2: PyTorch Native 原生对比法 (极速 FlashAttention 版)

对比:
   用户 Ragged 零 Padding (MOE-LLM.py 的 GroupTransformer, cu_seqlens 变长)
   vs
   PyTorch Native Dense Padding (官方 FlashAttention 后端 + is_causal=True)

核心优化: 彻底抛弃 4D 实体 Tensor Mask，改用 `attn_mask=None, is_causal=True`，
          强制解封 SDPA 的底层硬件融合 FlashAttention 内核！

测量: 端到端延迟(100轮) + 峰值显存 + TFLOPS + 无效FLOPs比例
"""

import importlib.util
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 ref_inference_llm.py 里"无 CPU-GPU 同步"的官方 MoE 层
from ref_inference_llm import RefMoE


# =====================================================================
# 1. 加载 MOE-LLM.py 的 GroupTransformer (不修改其原生代码)
# =====================================================================
def load_moe_module():
    spec = importlib.util.spec_from_file_location(
        "moe_llm", "/home/xkem/cuda_triton/project/MOE-LLM.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# 2. PyTorch Native Dense Padding 实现 (解封 FlashAttention 高效版)
# =====================================================================
class DensePaddingLLM(nn.Module):
    """输入 [B, max_len, d_model]，直接激活硬核 FlashAttention 后端。"""

    def __init__(self, d_model=512, num_heads=8, num_kv_heads=2, head_dim=64,
                 num_experts=8, top_k=2, d_ffn=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.input_norm = nn.RMSNorm(d_model, dtype=torch.float16)
        self.post_attn_norm = nn.RMSNorm(d_model, dtype=torch.float16)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=torch.float16)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=torch.float16)
        self.moe_layer = RefMoE(d_model, num_experts, top_k, d_ffn)

    def forward(self, x, seqlens=None):
        # x: [B, L, d_model] (Padding 到 max_len)
        B, L, D = x.shape
        H, KV, HD = self.num_heads, self.num_kv_heads, self.head_dim
        sm_scale = 1.0 / math.sqrt(HD)

        # 1. Pre-LN + QKV 投影
        x_norm = self.input_norm(x)
        q = self.q_proj(x_norm).view(B, L, H, HD).transpose(1, 2)   # [B, H, L, HD]
        k = self.k_proj(x_norm).view(B, L, KV, HD).transpose(1, 2)  # [B, KV, L, HD]
        v = self.v_proj(x_norm).view(B, L, KV, HD).transpose(1, 2)

        # 2. 官方 FlashAttention (GQA)
        # 【关键修改】：彻底取消 4D 实体 attn_mask，设为 None + is_causal=True
        # 激活极速硬件级 FlashAttention/MemEfficient CUDA 内核，拒绝退化到 Math 后端！
        o = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=True, 
            enable_gqa=True, 
            scale=sm_scale
        )  # [B, H, L, HD]

        o = o.transpose(1, 2).reshape(B, L, H * HD)
        o = self.out_proj(o)

        # 3. 残差 + MoE
        x = x + o
        residual = x
        x_norm = self.post_attn_norm(x)
        moe_out = self.moe_layer(x_norm.reshape(B * L, D)).reshape(B, L, D)
        x = residual + moe_out
        return x


# =====================================================================
# 3. 计时 / 显存 / FLOPs 工具
# =====================================================================
def measure(fn, n_rounds=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    t = torch.tensor(times)
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p50": torch.quantile(t, 0.5).item(),
        "p99": torch.quantile(t, 0.99).item(),
    }


def measure_peak_vram(fn):
    fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024 * 1024)


def compute_flops(seqlens, d_model, num_heads, num_kv_heads, head_dim,
                  top_k, d_ffn, num_experts):
    N = sum(seqlens)
    flops = 0.0
    flops += 2 * N * d_model * d_model
    flops += 2 * N * d_model * (num_kv_heads * head_dim) * 2
    flops += 2 * N * d_model * d_model
    for L in seqlens:
        flops += 2 * num_heads * head_dim * L * (L + 1)
    flops += N * 4 * top_k * d_model * d_ffn
    flops += 2 * N * d_model * num_experts
    return flops


def to_tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


# =====================================================================
# 4. 单场景对比
# =====================================================================
def bench_case(model_ragged, model_dense, seqlens, d_model, device, config):
    B = len(seqlens)
    max_len = max(seqlens)

    # ---- 相同的真实 token ----
    real_xs = [torch.randn(L, d_model, dtype=torch.float16, device=device) for L in seqlens]

    # ---- Ragged 输入 ----
    x_ragged = torch.cat(real_xs, dim=0)
    cu = [0]
    acc = 0
    for L in seqlens:
        acc += L
        cu.append(acc)
    cu_ragged = torch.tensor(cu, dtype=torch.int32, device=device)

    # ---- Dense Padding 输入 ----
    x_dense = torch.zeros(B, max_len, d_model, dtype=torch.float16, device=device)
    for b, x_s in enumerate(real_xs):
        x_dense[b, :x_s.shape[0]] = x_s
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=device)

    fn_ragged = lambda: model_ragged(x_ragged, cu_ragged, max_len)
    fn_dense = lambda: model_dense(x_dense, seqlens_t)

    # ---- 预热 ----
    for _ in range(10):
        fn_ragged()
        fn_dense()
    torch.cuda.synchronize()

    ragged = measure(fn_ragged)
    dense = measure(fn_dense)

    vram_ragged = measure_peak_vram(fn_ragged)
    vram_dense = measure_peak_vram(fn_dense)

    flops_ragged = compute_flops(
        seqlens, config["d_model"], config["num_heads"], config["num_kv_heads"],
        config["head_dim"], config["top_k"], config["d_ffn"], config["num_experts"],
    )
    flops_dense = compute_flops(
        [max_len] * B, config["d_model"], config["num_heads"], config["num_kv_heads"],
        config["head_dim"], config["top_k"], config["d_ffn"], config["num_experts"],
    )
    total_tokens = sum(seqlens)
    dense_tokens = B * max_len
    waste_ratio = (dense_tokens - total_tokens) / dense_tokens

    print(f"seqlens={seqlens}  B={B}  max_len={max_len}")
    print(f"  token 数: Ragged={total_tokens}  Dense={dense_tokens}  "
          f"无效 padding 比例={waste_ratio * 100:.1f}%")
    print(f"  --- 延迟 (100轮) ---")
    print(f"    Ragged(自写Triton): mean={ragged['mean']:.4f} ms  std={ragged['std']:.4f}  "
          f"p50={ragged['p50']:.4f}  p99={ragged['p99']:.4f}")
    print(f"    Dense (PyTorch原生): mean={dense['mean']:.4f} ms  std={dense['std']:.4f}  "
          f"p50={dense['p50']:.4f}  p99={dense['p99']:.4f}")
    print(f"    延迟加速比(mean) = {dense['mean'] / ragged['mean']:.2f}x")
    print(f"  --- TFLOPS (各自实际 FLOPs) ---")
    print(f"    Ragged: {to_tflops(flops_ragged, ragged['mean']):.3f}  "
          f"(有效 {flops_ragged / 1e9:.2f} GFLOPs)")
    print(f"    Dense : {to_tflops(flops_dense, dense['mean']):.3f}  "
          f"(全量 {flops_dense / 1e9:.2f} GFLOPs, 含无效 padding)")
    print(f"  --- 峰值显存 ---")
    print(f"    激活峰值: Ragged={vram_ragged:.2f} MB  Dense={vram_dense:.2f} MB")
    if vram_dense > 0:
        print(f"    显存节省(激活) = {(1 - vram_ragged / vram_dense) * 100:.1f}%")
    print()
    return {"speedup": dense["mean"] / ragged["mean"], "waste_ratio": waste_ratio}


# =====================================================================
# 5. 主流程
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    config_path = "/home/xkem/cuda_triton/project/parameter/model_config.json"
    weights_path = "/home/xkem/cuda_triton/project/parameter/model_weights.pt"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    d_model = config["d_model"]

    moe = load_moe_module()
    model_ragged = moe.GroupTransformer(**config).to(device)
    model_dense = DensePaddingLLM(**config).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model_ragged.load_state_dict(state_dict, strict=True)
    model_dense.load_state_dict(state_dict, strict=True)
    model_ragged.eval()
    model_dense.eval()
    print(f"✅ 两个模型权重加载成功 (strict=True, 同一份 model_weights.pt)")
    print(f"device={device}  d_model={d_model}  num_heads={config['num_heads']}  "
          f"num_kv_heads={config['num_kv_heads']}\n")

    cases = [
        ("长尾(真实 serving)", [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]),
        ("分桶(传统 Dense 最佳实践)", [1600, 1728, 1792, 1856, 1920, 1984, 2048]),
        ("极端短序列+长尾", [100, 100, 100, 100, 100, 100, 100, 100, 100, 2048]),
    ]

    print("========== Ragged(自写Triton) vs Dense Padding(PyTorch Native FlashAttention) ==========\n")
    for name, seqlens in cases:
        print(f"=== {name} ===")
        bench_case(model_ragged, model_dense, seqlens, d_model, device, config)