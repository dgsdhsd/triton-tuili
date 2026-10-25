#!/usr/bin/env python3
"""
方案1: Ragged Packed vs Dense Padding (假 Offset 模拟法)

用同一个 GroupTransformer kernel, 仅改变「输入 + offsets」来模拟 Dense Padding:
  - Ragged: 变长序列直接拼接, offsets = 实际累积长度 [0, L1, L1+L2, ...]
  - Dense : 每个序列 pad 0 到 max_len, offsets = 等长 [0, max_len, 2*max_len, ...]

说明: Dense 模拟组强制 GPU 计算包括 padding 在内的全部 B*max_len 个 token,
      FLOPs 与显存搬运量与真正的 Dense 实现等价 (仅性能对比, 非数值对比)。

测量: 端到端延迟(100轮 mean/std/p50/p99) + 峰值显存(Peak VRAM) + TFLOPS + 无效FLOPs比例
"""

import importlib.util
import json
import sys

import torch


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
# 2. 计时 / 显存 / FLOPs 工具
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
    """返回 forward 期间的峰值激活显存增量 (MB)。"""
    fn()  # 让缓存稳定
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # 清缓存, 使下一次 forward 重新分配中间激活
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()  # 冷缓存, 真实反映激活峰值
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024 * 1024)


def compute_flops(seqlens, d_model, num_heads, num_kv_heads, head_dim,
                  top_k, d_ffn, num_experts):
    """总 FLOPs (causal attention + QKV/out 投影 + MoE + router)。"""
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
# 3. 单场景对比
# =====================================================================
def bench_case(model, seqlens, d_model, device, config):
    B = len(seqlens)
    max_len = max(seqlens)

    # ---- 生成相同的真实 token ----
    real_xs = [torch.randn(L, d_model, dtype=torch.float16, device=device) for L in seqlens]

    # ---- Ragged: 直接拼接 ----
    x_ragged = torch.cat(real_xs, dim=0)
    cu_ragged = [0]
    acc = 0
    for L in seqlens:
        acc += L
        cu_ragged.append(acc)
    cu_ragged = torch.tensor(cu_ragged, dtype=torch.int32, device=device)

    # ---- Dense 模拟: 每个序列 pad 0 到 max_len ----
    padded = []
    for x_s, L in zip(real_xs, seqlens):
        if L < max_len:
            pad = torch.zeros(max_len - L, d_model, dtype=torch.float16, device=device)
            padded.append(torch.cat([x_s, pad], dim=0))
        else:
            padded.append(x_s)
    x_dense = torch.cat(padded, dim=0)
    cu_dense = torch.tensor([i * max_len for i in range(B + 1)], dtype=torch.int32, device=device)

    fn_ragged = lambda: model(x_ragged, cu_ragged, max_len)
    fn_dense = lambda: model(x_dense, cu_dense, max_len)

    # ---- 预热 ----
    for _ in range(10):
        fn_ragged()
        fn_dense()
    torch.cuda.synchronize()

    # ---- 延迟 (100 轮) ----
    ragged = measure(fn_ragged)
    dense = measure(fn_dense)

    # ---- 峰值显存 ----
    vram_ragged = measure_peak_vram(fn_ragged)
    vram_dense = measure_peak_vram(fn_dense)

    # ---- FLOPs 与浪费比例 ----
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

    # ---- 输入显存 (理论, fp16=2B/元素) ----
    input_vram_ragged = total_tokens * d_model * 2 / (1024 * 1024)
    input_vram_dense = dense_tokens * d_model * 2 / (1024 * 1024)

    # ---- 输出 ----
    print(f"seqlens={seqlens}  B={B}  max_len={max_len}")
    print(f"  token 数: Ragged={total_tokens}  Dense={dense_tokens}  "
          f"无效 padding 比例={waste_ratio * 100:.1f}%")
    print(f"  --- 延迟 (100轮) ---")
    print(f"    Ragged: mean={ragged['mean']:.4f} ms  std={ragged['std']:.4f}  "
          f"p50={ragged['p50']:.4f}  p99={ragged['p99']:.4f}")
    print(f"    Dense : mean={dense['mean']:.4f} ms  std={dense['std']:.4f}  "
          f"p50={dense['p50']:.4f}  p99={dense['p99']:.4f}")
    print(f"    延迟加速比(mean) = {dense['mean'] / ragged['mean']:.2f}x")
    print(f"  --- TFLOPS (各自实际 FLOPs) ---")
    print(f"    Ragged: {to_tflops(flops_ragged, ragged['mean']):.3f}  "
          f"(有效 {flops_ragged / 1e9:.2f} GFLOPs)")
    print(f"    Dense : {to_tflops(flops_dense, dense['mean']):.3f}  "
          f"(全量 {flops_dense / 1e9:.2f} GFLOPs, 含无效 padding)")
    print(f"  --- 峰值显存 ---")
    print(f"    输入显存: Ragged={input_vram_ragged:.2f} MB  Dense={input_vram_dense:.2f} MB")
    print(f"    激活峰值: Ragged={vram_ragged:.2f} MB  Dense={vram_dense:.2f} MB")
    if vram_dense > 0:
        print(f"    显存节省(激活) = {(1 - vram_ragged / vram_dense) * 100:.1f}%")
    print()
    return {
        "speedup": dense["mean"] / ragged["mean"],
        "waste_ratio": waste_ratio,
        "vram_ragged": vram_ragged,
        "vram_dense": vram_dense,
    }


# =====================================================================
# 4. 主流程
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
    model = moe.GroupTransformer(**config).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"device={device}  d_model={d_model}  num_heads={config['num_heads']}  "
          f"num_kv_heads={config['num_kv_heads']}\n")

    # 三个场景: 长尾 / 分桶 / 短序列(等长)
    cases = [
        ("长尾(真实 serving)", [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]),
        ("分桶(传统 Dense 最佳实践)", [1600, 1728, 1792, 1856, 1920, 1984, 2048]),
        ("极端短序列+长尾", [100, 100, 100, 100, 100, 100, 100, 100, 100, 2048]),
    ]

    print("========== Ragged Packed vs Dense Padding (假 Offset 模拟) ==========\n")
    for name, seqlens in cases:
        print(f"=== {name} ===")
        bench_case(model, seqlens, d_model, device, config)

