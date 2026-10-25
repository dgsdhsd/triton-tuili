#!/usr/bin/env python3
"""
性能对比: 多序列 ragged 一次处理  vs  单序列循环多次调用

模式 A (多组/batched):
    MOE-LLM.py 的 GroupTransformer, 一次输入多个序列 (ragged, cu_seqlens), 一次 kernel 调用。
模式 B (单组/逐序列):
    ref_inference_llm.py 的 RefInferenceLLM (官方函数), 一次只能跑一个序列, 循环多次调用。

两者 GQA 结构一致 (8 Q heads / 2 KV heads), 权重一致 (同一份 model_weights.pt)。
目标: 量化"多序列一次处理"相比"单序列多次运行"能快多少。
"""

import importlib.util
import json
import math
import sys

import torch
import triton


# =====================================================================
# 1. 加载两个模型
# =====================================================================
def load_moe_module():
    spec = importlib.util.spec_from_file_location(
        "moe_llm", "/home/xkem/cuda_triton/project/MOE-LLM.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_models(config, weights_path, device):
    moe = load_moe_module()
    from ref_inference_llm import RefInferenceLLM
    #为什么我的不行，因为要求改成MOE_LLM.py才行

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    model_multi = moe.GroupTransformer(**config).to(device)
    model_multi.load_state_dict(state_dict, strict=True)
    model_multi.eval()

    model_single = RefInferenceLLM(**config).to(device)
    model_single.load_state_dict(state_dict, strict=True)
    model_single.eval()
    return model_multi, model_single


# =====================================================================
# 2. 构造数据
# =====================================================================
def make_data(seqlens, d_model, device):
    """返回 模式A 的整体输入 与 模式B 的逐序列列表 (同一份数据)。"""
    xs = [torch.randn(L, d_model, dtype=torch.float16, device=device) for L in seqlens]
    x_all = torch.cat(xs, dim=0)

    cu = [0]
    acc = 0
    for L in seqlens:
        acc += L
        cu.append(acc)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    max_seqlen = max(seqlens)
    return x_all, cu_seqlens, max_seqlen, xs


# =====================================================================
# 3. 计时 / FLOPs 计算 / Benchmark
# =====================================================================
def measure(fn, n_rounds=100, warmup=10):
    """用 CUDA Event 计时 n_rounds 轮, 返回各项统计 (单位 ms)。"""
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


def compute_flops(seqlens, d_model, num_heads, num_kv_heads, head_dim,
                  top_k, d_ffn, num_experts):
    """总有效 FLOPs (causal attention + QKV/out 投影 + MoE + router)。

    口径:
      - QKV 投影:  Q [N,D]@[D,D] + K/V [N,D]@[D, kv_heads*D]
      - out 投影:  [N,D]@[D,D]
      - attention: 因果三角, 每个序列 QK^T + PV 合计 2*H*D*L*(L+1)
      - MoE:       每 token top_k 专家, up [D]@[D,d_ffn] + down [d_ffn]@[d_ffn,D]
      - router:    [N,D]@[D,num_experts]
      (RMSNorm / softmax / SiLU 等非线性不计入, 两模式相同, 不影响对比)
    """
    N = sum(seqlens)
    flops = 0.0
    flops += 2 * N * d_model * d_model                                 # Q proj
    flops += 2 * N * d_model * (num_kv_heads * head_dim) * 2          # K, V proj
    flops += 2 * N * d_model * d_model                                 # out proj
    for L in seqlens:
        flops += 2 * num_heads * head_dim * L * (L + 1)               # causal attention
    flops += N * 4 * top_k * d_model * d_ffn                          # MoE up+down
    flops += 2 * N * d_model * num_experts                            # router
    return flops


def to_tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


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


def bench_case(model_multi, model_single, seqlens, d_model, device, config):
    x_all, cu_seqlens, max_seqlen, xs = make_data(seqlens, d_model, device)
    B = len(seqlens)
    total_tokens = sum(seqlens)

    # 模式 A: 多序列一次处理
    def fn_multi():
        with torch.no_grad():
            return model_multi(x_all, cu_seqlens, max_seqlen)

    # 模式 B: 单序列循环多次
    def fn_loop():
        with torch.no_grad():
            outs = []
            for x_s in xs:
                outs.append(model_single(x_s))
            return torch.cat(outs, dim=0)

    # 预热 (触发 autotune / cuBLAS / cuDNN 初始化)
    for _ in range(10):
        fn_multi()
        fn_loop()
    torch.cuda.synchronize()

    multi = measure(fn_multi)
    loop = measure(fn_loop)
    vram_multi = measure_peak_vram(fn_multi)
    vram_loop = measure_peak_vram(fn_loop)

    flops = compute_flops(
        seqlens, config["d_model"], config["num_heads"], config["num_kv_heads"],
        config["head_dim"], config["top_k"], config["d_ffn"], config["num_experts"],
    )
    speedup = loop["mean"] / multi["mean"]

    print(f"seqlens={seqlens}  B={B}  max_len={max_seqlen}")
    print(f"  token 数: 总={total_tokens} (两模式处理相同 token)")
    print(f"  --- 延迟 (100轮) ---")
    print(f"    多发射一次(用户kernel): mean={multi['mean']:.4f} ms  std={multi['std']:.4f}  "
          f"p50={multi['p50']:.4f}  p99={multi['p99']:.4f}")
    print(f"    单发射循环(官方函数):   mean={loop['mean']:.4f} ms  std={loop['std']:.4f}  "
          f"p50={loop['p50']:.4f}  p99={loop['p99']:.4f}")
    print(f"    延迟加速比(mean) = {speedup:.2f}x")
    print(f"  --- TFLOPS (有效 FLOPs) ---")
    print(f"    多发射: {to_tflops(flops, multi['mean']):.3f}  (有效 {flops / 1e9:.2f} GFLOPs)")
    print(f"    单发射: {to_tflops(flops, loop['mean']):.3f}  (有效 {flops / 1e9:.2f} GFLOPs)")
    print(f"  --- 峰值显存 ---")
    print(f"    激活峰值: 多发射={vram_multi:.2f} MB  单发射={vram_loop:.2f} MB")
    print()
    return speedup


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

    model_multi, model_single = build_models(config, weights_path, device)
    print(f"device={device}  d_model={d_model}  num_heads={config['num_heads']}  "
          f"num_kv_heads={config['num_kv_heads']}\n")

    cases = [
        ("12 个短序列",   [32] * 12),
        ("12 个长序列",   [2000] * 12),
        ("12 个中等序列", [512] * 12),
    ]

    print("========== 多序列一次 vs 单序列循环 ==========\n")
    for name, seqlens in cases:
        print(f"--- {name} ---")
        bench_case(model_multi, model_single, seqlens, d_model, device, config)
        print()
