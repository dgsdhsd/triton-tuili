#!/usr/bin/env python3
"""
验证 FlashAttentionWrapper 的正确性 (自写 Triton kernel vs 官方 FlashAttention)

对比基准:
    torch.nn.functional.scaled_dot_product_attention (PyTorch 官方, 后端为 cuDNN FlashAttention)

方法:
    本 kernel 支持 GQA(多 group) 与变长序列(ragged)。
    参考实现按 cu_seqlens 把每个序列切片, 对每个序列单独跑一次 GQA 的官方 SDPA,
    再拼接回 [N, H, D] 与 kernel 输出对比 —— 即"多次运行单组官方 attention 再拼接"。

权重对齐:
    q/k/v 不是随机生成, 而是用 project/parameter/model_weights.pt 里的真实投影权重
    (q_proj / k_proj / v_proj) + input_norm 计算得到, 保证 W 与真实模型对齐。

验证项:
    1. causal=True  (IS_CAUSAL, 走 Stage1/Stage2 分带)
    2. causal=False (dense 模式, 走 STAGE=1 全量)
    3. 多组边界 seqlens (64 对齐/非对齐, 极短, 长序列, 变长)
"""

import importlib.util
import math
import sys

import torch
import torch.nn.functional as F

# =====================================================================
# 1. 复用 MOE-LLM.py 里的 FlashAttentionWrapper / GroupTransformer 定义
# =====================================================================
def load_moe_module():
    path = "/home/xkem/cuda_triton/project/MOE-LLM.py"
    spec = importlib.util.spec_from_file_location("moe_llm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


moe = load_moe_module()
FlashAttentionWrapper = moe.FlashAttentionWrapper
GroupTransformer = moe.GroupTransformer
load_model_config = moe.load_model_config
load_model_weights = moe.load_model_weights


# =====================================================================
# 2. 官方 reference: GQA + 变长 (多次运行单组官方 attention 再拼接)
# =====================================================================
def ref_gqa_attention(q, k, v, cu_seqlens, sm_scale, causal):
    """q: [N, H, D]  k/v: [N, KV, D]  ->  [N, H, D]

    对 cu_seqlens 划分出的每个序列独立调用官方 SDPA(GQA), 结果拼接。
    """
    N, H, D = q.shape
    outs = []
    for s in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[s].item())
        end = int(cu_seqlens[s + 1].item())
        if end <= start:
            continue
        # [L, H, D] -> [1, H, L, D]
        qs = q[start:end].unsqueeze(0).transpose(1, 2)
        ks = k[start:end].unsqueeze(0).transpose(1, 2)
        vs = v[start:end].unsqueeze(0).transpose(1, 2)

        o = F.scaled_dot_product_attention(
            qs, ks, vs,
            enable_gqa=True,       # GQA: 8 Q heads 共享 2 KV heads
            is_causal=causal,
            scale=sm_scale,
        )  # [1, H, L, D]

        outs.append(o.transpose(1, 2).squeeze(0))  # [L, H, D]
    return torch.cat(outs, 0)


# =====================================================================
# 3. 对比与报告
# =====================================================================
def report(name, mine, ref, atol=1e-2, rtol=1e-2):
    mine = mine.float()
    ref = ref.float()
    abs_diff = (mine - ref).abs()
    denom = ref.abs().clamp_min(1e-6)
    rel_diff = abs_diff / denom

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    ok = torch.allclose(mine, ref, atol=atol, rtol=rtol)

    print(
        f"[{name}]\n"
        f"    max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}  "
        f"max_rel={max_rel:.6e}  mean_rel={mean_rel:.6e}\n"
        f"    allclose(atol={atol}, rtol={rtol}) = {ok}"
    )
    return ok


def run_case(model, seqlens, causal, device, head_dim):
    cu_list = [0]
    acc = 0
    for L in seqlens:
        acc += L
        cu_list.append(acc)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
    max_seqlen = max(seqlens)
    N = cu[-1].item()

    d_model = model.d_model
    num_heads = model.num_heads
    num_kv_heads = model.num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)

    x = torch.randn(N, d_model, dtype=torch.float16, device=device)

    with torch.no_grad():
        # ---- 用真实投影权重生成 q/k/v (W 对齐 model_weights.pt) ----
        x_norm = model.input_norm(x)
        q = model.q_proj(x_norm).view(N, num_heads, head_dim)
        k = model.k_proj(x_norm).view(N, num_kv_heads, head_dim)
        v = model.v_proj(x_norm).view(N, num_kv_heads, head_dim)

        # ---- 自写 Triton kernel ----
        mine = FlashAttentionWrapper.apply(q, k, v, cu, max_seqlen, causal, None, True)

    # ---- 官方 reference ----
    ref = ref_gqa_attention(q, k, v, cu, sm_scale, causal)

    return report(f"seqlens={seqlens} causal={causal}", mine, ref)



# =====================================================================
# 4. 主流程
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    config_path = "/home/xkem/cuda_triton/project/parameter/model_config.json"
    weights_path = "/home/xkem/cuda_triton/project/parameter/model_weights.pt"

    config = load_model_config(config_path)
    model = GroupTransformer(**config).to(device)
    load_model_weights(model, weights_path)
    model.eval()

    head_dim = config["head_dim"]
    print(f"device={device}  d_model={config['d_model']}  "
          f"num_heads={config['num_heads']}  num_kv_heads={config['num_kv_heads']}  "
          f"head_dim={head_dim}\n")

    # ---- 测试用例 ----
    causal_cases = [
        [60, 100, 250, 480, 950, 2000],   # 与主程序一致
        [64, 64, 64, 64],                 # 全是 64 的倍数
        [63, 64, 65, 127, 128, 129],      # 64 边界附近 (ragged 尾部)
        [1, 2, 3, 4, 5, 6],               # 极短序列
        [7],                              # 单短序列
        [1000],                           # 单长序列
        [1],                              # 最小序列
    ]
    non_causal_cases = [
        [60, 100, 250, 480, 950, 2000],   # dense 模式主用例
        [63, 64, 65],
    ]

    results = []
    print("========== causal=True (Stage1/Stage2) ==========")
    for seqlens in causal_cases:
        results.append(run_case(model, seqlens, True, device, head_dim))

    print("\n========== causal=False (dense) ==========")
    for seqlens in non_causal_cases:
        results.append(run_case(model, seqlens, False, device, head_dim))

    print("\n========== 汇总 ==========")
    print(f"通过 {sum(results)} / {len(results)} 用例")
    if all(results):
        print("✅ 全部通过")
    else:
        print("❌ 存在失败用例")
