"""验证 _paged_attn_decode_kernel 的逻辑准确性。

场景：decode 阶段，每个序列 1 个新 token，需要：
  1. 把新 token 的 K/V 落盘到 paged cache（去重：每个 KV 组第一个 head 写一次）；
  2. 读历史 cache + 新 token，做因果注意力（新 token 能看到所有历史 + 自己）。

对比基准：手写分序列注意力（cat(历史, 新) 后做 softmax），独立于内核。

运行：
  cd /home/xkem/cuda_triton
  vllm_env/bin/python project/models/_paged_attn_decode_validate.py
"""
import importlib.util
import math
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "self_llm", "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["self_llm"] = mod
spec.loader.exec_module(mod)
kernel = mod._paged_attn_decode_kernel

H, KVH, D = 8, 2, 64
BS = 16  # BLOCK_SIZE
SM_SCALE = 1.0 / math.sqrt(D)
GROUP = H // KVH


def reference_decode(q, k_new, v_new, k_hist, v_hist, context_lens):
    """分序列参考：新 token 对 [历史 + 自己] 做因果注意力。"""
    S, _, _ = q.shape
    outs = []
    for s in range(S):
        ctx = int(context_lens[s])
        k_full = torch.cat([k_hist[s], k_new[s:s + 1]], dim=0)  # [ctx, KVH, D]
        v_full = torch.cat([v_hist[s], v_new[s:s + 1]], dim=0)
        k_full = k_full.repeat_interleave(GROUP, dim=1)  # GQA 广播到 H 个头
        v_full = v_full.repeat_interleave(GROUP, dim=1)
        scores = torch.einsum('hd,khd->hk', q[s].float(), k_full.float()) * SM_SCALE
        p = torch.softmax(scores, dim=-1)  # [H, ctx]
        out = torch.einsum('hk,khd->hd', p, v_full.float()).half()
        outs.append(out)
    return torch.stack(outs, dim=0)  # [S, H, D]


def run():
    torch.manual_seed(0)
    S = 3
    context_lens = torch.tensor([5, 8, 3], dtype=torch.int32, device="cuda")  # 含新 token
    prefix_lens = [int(c) - 1 for c in context_lens]  # 历史长度

    q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    k_new = torch.randn(S, KVH, D, device="cuda", dtype=torch.float16)
    v_new = torch.randn(S, KVH, D, device="cuda", dtype=torch.float16)

    k_hist = [torch.randn(p, KVH, D, device="cuda", dtype=torch.float16) for p in prefix_lens]
    v_hist = [torch.randn(p, KVH, D, device="cuda", dtype=torch.float16) for p in prefix_lens]

    # 构建 paged cache 与 block_table（历史 + 新 token 的 slot 都连续排布）
    blocks_per_seq = [max(1, (int(c) + BS - 1) // BS) for c in context_lens]
    starts = [0] + torch.cumsum(torch.tensor(blocks_per_seq, dtype=torch.int64), 0).tolist()
    num_blocks = starts[-1]
    max_blocks = max(blocks_per_seq)

    block_tables = torch.zeros(S, max_blocks, dtype=torch.int32, device="cuda")
    K_cache = torch.zeros(num_blocks, KVH, BS, D, dtype=torch.float16, device="cuda")
    V_cache = torch.zeros(num_blocks, KVH, BS, D, dtype=torch.float16, device="cuda")
    slot_mapping = torch.zeros(S, dtype=torch.int32, device="cuda")

    for s in range(S):
        bs = starts[s]
        block_tables[s, :blocks_per_seq[s]] = torch.arange(bs, starts[s + 1], dtype=torch.int32, device="cuda")
        # 写历史
        for pos in range(prefix_lens[s]):
            lb, ob = pos // BS, pos % BS
            K_cache[bs + lb, :, ob, :] = k_hist[s][pos]
            V_cache[bs + lb, :, ob, :] = v_hist[s][pos]
        # 新 token 的 slot（历史之后的第一个空位）
        abs_pos = prefix_lens[s]
        lb, ob = abs_pos // BS, abs_pos % BS
        slot_mapping[s] = (bs + lb) * BS + ob

    # ---- 调用 kernel（KV 落盘 + 读历史 cache + 读新 token from input 都在核内完成）----
    out = torch.empty_like(q)
    grid = (S, H)
    kernel[grid](
        Q=q, K_new=k_new, V_new=v_new, Out=out,
        K_cache=K_cache, V_cache=V_cache,
        block_tables=block_tables, slot_mapping=slot_mapping,
        context_lens=context_lens,
        sm_scale=SM_SCALE,
        stride_q_seq=q.stride(0), stride_q_head=q.stride(1), stride_q_dim=q.stride(2),
        stride_knew_seq=k_new.stride(0), stride_knew_head=k_new.stride(1), stride_knew_dim=k_new.stride(2),
        stride_vnew_seq=v_new.stride(0), stride_vnew_head=v_new.stride(1), stride_vnew_dim=v_new.stride(2),
        stride_o_seq=out.stride(0), stride_o_head=out.stride(1), stride_o_dim=out.stride(2),
        stride_block_table_seq=block_tables.stride(0),
        stride_k_block=K_cache.stride(0), stride_k_head=K_cache.stride(1),
        stride_k_bs=K_cache.stride(2), stride_k_dim=K_cache.stride(3),
        stride_v_block=V_cache.stride(0), stride_v_head=V_cache.stride(1),
        stride_v_bs=V_cache.stride(2), stride_v_dim=V_cache.stride(3),
        NUM_HEADS=H, NUM_KV_HEADS=KVH, BLOCK_SIZE=BS, BLOCK_D=D,
        num_warps=4, num_stages=2,
    )
    torch.cuda.synchronize()

    ref = reference_decode(q, k_new, v_new, k_hist, v_hist, context_lens)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-6)
    print(f"[decode] context_lens={context_lens.tolist()}")
    print(f"   max_abs_err = {diff.max().item():.6f}   mean_abs_err = {diff.mean().item():.6f}")
    print(f"   max_rel_err = {rel.max().item():.6f}   out_has_NaN = {torch.isnan(out).any().item()}")

    # 校验 KV 落盘：scatter 已把新 token 写进 cache（此处为验证用，实际 forward 同逻辑）
    write_ok = True
    for s in range(S):
        abs_pos = prefix_lens[s]
        lb, ob = abs_pos // BS, abs_pos % BS
        bs = starts[s]
        if (K_cache[bs + lb, :, ob, :] - k_new[s]).abs().max().item() > 1e-3:
            write_ok = False
        if (V_cache[bs + lb, :, ob, :] - v_new[s]).abs().max().item() > 1e-3:
            write_ok = False
    print(f"   cache_write_ok = {write_ok}")

    ok = diff.max().item() < 0.05 and write_ok and not torch.isnan(out).any().item()
    print(f"   => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    ok = run()
    print("=" * 70)
    print("PASS" if ok else "FAIL")
