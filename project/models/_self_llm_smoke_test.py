"""SELF-LLM.py 冒烟测试：验证「attention 空运行 + 其余链路」能否跑通。"""
import importlib.util
import sys
import torch

spec = importlib.util.spec_from_file_location(
    "self_llm", "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["self_llm"] = mod
spec.loader.exec_module(mod)

print("[1] 导入成功")

# ---------------- 1. 直接测 FlashAttentionWrapper 空运行 ----------------
fa = mod.FlashAttentionWrapper(head_dim=64, layer_name="test_attn")
print("[2] FlashAttentionWrapper.dry_run =", fa.dry_run)

N, H, KVH, D = 13, 8, 2, 64
q = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(N, KVH, D, device="cuda", dtype=torch.float16)
v = torch.randn(N, KVH, D, device="cuda", dtype=torch.float16)

out = fa(q, k, v, kv_cache=None, causal=True)
assert out.shape == (N, H, D), f"shape 错误: {out.shape}"
assert out.dtype == torch.float16, f"dtype 错误: {out.dtype}"
print("[3] FlashAttentionWrapper 空运行 forward OK, out.shape =", tuple(out.shape))

# ---------------- 2. 测 GroupTransformer 完整 forward (含 MoE) ----------------
gt = mod.GroupTransformer(
    d_model=512, num_heads=8, num_kv_heads=2, head_dim=64,
    num_experts=8, top_k=2, d_ffn=2048,
).cuda()
gt.eval()
print("[4] GroupTransformer 实例化 OK")

hidden = torch.randn(N, 512, device="cuda", dtype=torch.float16)
positions = torch.arange(N, device="cuda", dtype=torch.long)
with torch.no_grad():
    out = gt(hidden, positions)
assert out.shape == (N, 512), f"shape 错误: {out.shape}"
print("[5] GroupTransformer.forward (attention 空运行 + MoE) OK, out.shape =", tuple(out.shape))

print("\n✅ 冒烟测试全部通过")
