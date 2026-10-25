"""验证 _ragged_attn_fwd_kernel_persistent 的逻辑准确性。

覆盖场景:
  1. 纯 prefill (prefix_len=0): Stage A 跳过, 只算新 token 的因果注意力。
  2. prefix-prefill (prefix_len>0): Stage A 读历史 KV cache + Stage B 新 token。
     历史对新 token 全可见, 新 token 之间因果。

对比基准: 手写分序列注意力 + 显式绝对位置因果 mask (不依赖 paged 寻址, 独立于内核)。

运行:
  cd /home/xkem/cuda_triton
  vllm_env/bin/python project/models/_ragged_attn_validate.py
"""
import importlib.util
import math
import sys

import torch

# ---------------- 加载 SELF-LLM.py 里的内核 ----------------
spec = importlib.util.spec_from_file_location(
    "self_llm", "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["self_llm"] = mod
spec.loader.exec_module(mod)
kernel = mod._ragged_attn_fwd_kernel_persistent

H, KVH, D = 8, 2, 64
BS = 16          # BLOCK_SIZE: 每页 token 数
BM = BN = 64     # BLOCK_M / BLOCK_N
SM_SCALE = 1.0 / math.sqrt(D)


def reference_ragged(q, k, v, k_hist_list, v_hist_list,
                     cu_seqlens, context_lens, causal=True):
    """分序列参考注意力。q/k/v 为新 token 的张量, k_hist/v_hist 为每序列历史。"""
    T, H, D = q.shape
    KVH = k.shape[1]
    group = H // KVH
    S = cu_seqlens.numel() - 1
    outs = []
    for s in range(S):
        start = int(cu_seqlens[s])
        new_len = int(cu_seqlens[s + 1]) - start
        total_len = int(context_lens[s])
        prefix_len = total_len - new_len

        q_s = q[start:start + new_len]                    # [new_len, H, D]
        k_hist = k_hist_list[s]                           # [prefix_len, KVH, D]
        v_hist = v_hist_list[s]
        k_new = k[start:start + new_len]                  # [new_len, KVH, D]
        v_new = v[start:start + new_len]

        k_full = torch.cat([k_hist, k_new], dim=0) if prefix_len else k_new
        v_full = torch.cat([v_hist, v_new], dim=0) if prefix_len else v_new

        # GQA: KV 广播到 H 个头
        k_full = k_full.repeat_interleave(group, dim=1)   # [total_len, H, D]
        v_full = v_full.repeat_interleave(group, dim=1)

        # 绝对位置因果 mask
        q_abs = torch.arange(prefix_len, total_len, device=q.device)
        k_abs = torch.arange(0, total_len, device=q.device)

        scores = torch.einsum('qhd,khd->qkh', q_s.float(), k_full.float()) * SM_SCALE
        if causal:
            mask = (q_abs[:, None] >= k_abs[None, :])     # [new_len, total_len]
            scores = scores.masked_fill(~mask[:, :, None], float('-inf'))
        p = torch.softmax(scores, dim=1)                  # [new_len, total_len, H]
        out_s = torch.einsum('qkh,khd->qhd', p, v_full.float()).half()
        outs.append(out_s)
    return torch.cat(outs, dim=0)                         # [T, H, D]


def build_paged_cache(k_hist_list, v_hist_list, context_lens):
    """把每序列历史 K/V 写入分页缓存, 返回 K_cache/V_cache/block_tables/starts。"""
    S = len(context_lens)
    blocks_per_seq = [max(1, (int(l) + BS - 1) // BS) for l in context_lens]
    starts = [0] + torch.cumsum(
        torch.tensor(blocks_per_seq, dtype=torch.int64), 0).tolist()
    num_blocks = starts[-1]
    max_blocks = max(blocks_per_seq)

    block_tables = torch.zeros(S, max_blocks, dtype=torch.int32, device='cuda')
    K_cache = torch.zeros(num_blocks, KVH, BS, D, dtype=torch.float16, device='cuda')
    V_cache = torch.zeros(num_blocks, KVH, BS, D, dtype=torch.float16, device='cuda')

    for s in range(S):
        bs = starts[s]
        block_tables[s, :blocks_per_seq[s]] = torch.arange(
            bs, starts[s + 1], dtype=torch.int32, device='cuda')
        k_hist = k_hist_list[s]
        prefix_len = k_hist.shape[0]
        for pos in range(prefix_len):
            lb, ob = pos // BS, pos % BS
            K_cache[bs + lb, :, ob, :] = k_hist[pos]
            V_cache[bs + lb, :, ob, :] = v_hist_list[s][pos]
    return K_cache, V_cache, block_tables, blocks_per_seq, starts


def build_slot_mapping(cu_seqlens, context_lens, blocks_per_seq, starts):
    """给新 token 生成 slot_mapping (block*BS + offset)。"""
    T = int(cu_seqlens[-1])
    slot = torch.zeros(T, dtype=torch.int32, device='cuda')
    S = cu_seqlens.numel() - 1
    for s in range(S):
        start = int(cu_seqlens[s])
        new_len = int(cu_seqlens[s + 1]) - start
        prefix_len = int(context_lens[s]) - new_len
        bs = starts[s]
        for i in range(new_len):
            abs_pos = prefix_len + i
            lb, ob = abs_pos // BS, abs_pos % BS
            slot[start + i] = (bs + lb) * BS + ob
    return slot


def run_scenario(name, new_lens, prefix_lens, causal=True):
    torch.manual_seed(0)
    S = len(new_lens)
    T = sum(new_lens)
    context_lens = [new_lens[i] + prefix_lens[i] for i in range(S)]
    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(torch.tensor(new_lens, dtype=torch.int32), 0).tolist(),
        dtype=torch.int32, device='cuda')
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device='cuda')

    # 新 token 的 Q/K/V
    q = torch.randn(T, H, D, device='cuda', dtype=torch.float16)
    k = torch.randn(T, KVH, D, device='cuda', dtype=torch.float16)
    v = torch.randn(T, KVH, D, device='cuda', dtype=torch.float16)

    # 每序列历史 K/V
    k_hist_list = [torch.randn(p, KVH, D, device='cuda', dtype=torch.float16)
                   for p in prefix_lens]
    v_hist_list = [torch.randn(p, KVH, D, device='cuda', dtype=torch.float16)
                   for p in prefix_lens]

    K_cache, V_cache, block_tables, blocks_per_seq, starts = build_paged_cache(
        k_hist_list, v_hist_list, context_lens)
    slot_mapping = build_slot_mapping(cu_seqlens, context_lens, blocks_per_seq, starts)

    # tile 映射
    new_lens_t = torch.tensor(new_lens, dtype=torch.int32, device='cuda')
    tiles_per_seq = (new_lens_t + BM - 1) // BM
    cu_tiles = torch.zeros(S + 1, dtype=torch.int32, device='cuda')
    cu_tiles[1:] = torch.cumsum(tiles_per_seq, dim=0)
    total_tiles = int(cu_tiles[-1])
    total_tasks = total_tiles * H
    tile2seq = torch.repeat_interleave(
        torch.arange(S, dtype=torch.int32, device='cuda'),
        tiles_per_seq.to(torch.int64))
    tile_counter = torch.zeros(1, dtype=torch.int32, device='cuda')

    out = torch.empty_like(q)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (min(num_sms, total_tasks),)

    kernel[grid](
        Q=q, K=k, V=v, Out=out,
        K_cache=K_cache, V_cache=V_cache,
        block_tables=block_tables, slot_mapping=slot_mapping,
        cu_tiles=cu_tiles, cu_seqlens=cu_seqlens, context_lens=context_lens_t,
        tile2seq_ptr=tile2seq, tile_counter_ptr=tile_counter,
        sm_scale=SM_SCALE,
        stride_qz=0, stride_qm=q.stride(0), stride_qh=q.stride(1),
        stride_kz=0, stride_km=k.stride(0), stride_kh=k.stride(1),
        stride_vz=0, stride_vm=v.stride(0), stride_vh=v.stride(1),
        stride_oz=0, stride_om=out.stride(0), stride_oh=out.stride(1),
        stride_block_table_seq=block_tables.stride(0),
        stride_k_block=K_cache.stride(0), stride_k_head=K_cache.stride(1),
        stride_k_bs=K_cache.stride(2), stride_k_dim=K_cache.stride(3),
        stride_v_block=V_cache.stride(0), stride_v_head=V_cache.stride(1),
        stride_v_bs=V_cache.stride(2), stride_v_dim=V_cache.stride(3),
        total_tasks=total_tasks,
        NUM_HEADS=H, NUM_KV_HEADS=KVH, BLOCK_SIZE=BS,
        IS_CAUSAL=causal,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=D,
        num_warps=4, num_stages=2,
    )
    torch.cuda.synchronize()

    ref = reference_ragged(q, k, v, k_hist_list, v_hist_list,
                           cu_seqlens, context_lens_t, causal)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-6)
    print(f"[{name}] new_lens={new_lens} prefix_lens={prefix_lens}")
    print(f"   max_abs_err = {diff.max().item():.6f}   "
          f"mean_abs_err = {diff.mean().item():.6f}")
    print(f"   max_rel_err = {rel.max().item():.6f}   "
          f"out_has_NaN = {torch.isnan(out).any().item()}")

    # 校验 cache 写回: 新 token 的 K/V 应被正确写入分页缓存
    write_ok = True
    for s in range(S):
        start = int(cu_seqlens[s])
        new_len = int(cu_seqlens[s + 1]) - start
        prefix_len = int(context_lens[s]) - new_len
        bs = starts[s]
        for i in range(new_len):
            abs_pos = prefix_len + i
            lb, ob = abs_pos // BS, abs_pos % BS
            if (K_cache[bs + lb, :, ob, :] - k[start + i]).abs().max().item() > 1e-3:
                write_ok = False
                break
    print(f"   cache_write_ok = {write_ok}")

    ok = diff.max().item() < 0.05 and write_ok
    print(f"   => {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    r1 = run_scenario("场景1 纯 prefill", new_lens=[100, 37], prefix_lens=[0, 0])
    r2 = run_scenario("场景2 prefix-prefill", new_lens=[50, 60], prefix_lens=[45, 20])
    r3 = run_scenario("场景3 大 prefix", new_lens=[13, 7], prefix_lens=[200, 33])
    r4 = run_scenario("场景4 非因果(双向)", new_lens=[40, 25], prefix_lens=[30, 10], causal=False)
    print("=" * 70)
    print("ALL PASS" if all([r1, r2, r3, r4]) else "SOME FAILED")
