"""验证 MoEInferenceLayer（Router + TopK + TMA Grouped GEMM + SiLU）的数值正确性。

对比基准：纯 PyTorch 逐专家逐 token 的参考实现（不依赖 Triton 内核）。

运行：
  cd /home/xkem/cuda_triton
  vllm_env/bin/python project/models/_moe_validate.py
"""
import importlib.util
import sys

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "self_llm", "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["self_llm"] = mod
spec.loader.exec_module(mod)
MoEInferenceLayer = mod.MoEInferenceLayer


def reference_moe(x, router_w, w1, w2, top_k):
    """纯 PyTorch 参考实现（逐 token、逐专家）。"""
    N, d_model = x.shape
    router_logits = F.linear(x.float(), router_w.float())  # [N, E]
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    out = torch.zeros(N, d_model, device=x.device, dtype=torch.float32)
    for n in range(N):
        for k in range(top_k):
            e = int(topk_indices[n, k])
            w = topk_weights[n, k]
            h = F.silu(F.linear(x[n].float(), w1[e].float()))  # [d_ffn]
            o = F.linear(h, w2[e].float())                      # [d_model]
            out[n] += w * o
    return out.half()


def run_once(N, d_model, num_experts, top_k, d_ffn, seed):
    torch.manual_seed(seed)
    moe = MoEInferenceLayer(d_model, num_experts, top_k, d_ffn).cuda()
    with torch.no_grad():
        # 模拟 dummy 权重：小量级随机初始化（避免 fp16 溢出）
        moe.router.weight.copy_(torch.randn_like(moe.router.weight.float()) * 0.02)
        moe.w1.copy_(torch.randn_like(moe.w1.float()) * 0.02)
        moe.w2.copy_(torch.randn_like(moe.w2.float()) * 0.02)

    x = torch.randn(N, d_model, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        out = moe(x)  # [N, d_model] fp16

    ref = reference_moe(x, moe.router.weight, moe.w1, moe.w2, top_k)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    print(f"[N={N}] max_abs_err={diff.max().item():.6f} mean_abs_err={diff.mean().item():.6f} "
          f"max_rel_err={rel.max().item():.6f} out_has_NaN={torch.isnan(out).any().item()}")
    ok = diff.max().item() < 0.02 and not torch.isnan(out).any().item()
    return ok


if __name__ == "__main__":
    print("=" * 70)
    # 用与 config.json 一致的规模：d_model=512, num_experts=8, top_k=2, d_ffn=1024
    results = []
    # 不同 token 数（13 是 prefill 的典型，2 是 decode 的典型，1 是单 token）
    for N in [1, 2, 13, 37]:
        results.append(run_once(N, 512, 8, 2, 1024, seed=N))
    print("=" * 70)
    print("ALL PASS" if all(results) else "SOME FAILED")
