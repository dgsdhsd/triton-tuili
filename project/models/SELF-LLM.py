from typing import Optional
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention import Attention
#======================================================================
#核函数列表
#======================================================================
@triton.jit
def _ragged_attn_fwd_kernel_persistent(
    Q, K, V, Out,
    # === Paged KV 物理池与寻址表 ===
    K_cache, V_cache, 
    block_tables, slot_mapping,
    # === 序列长度控制 ===
    cu_tiles, cu_seqlens, context_lens,
    tile2seq_ptr, tile_counter_ptr,
    sm_scale,
    # === 张量 Stride ===
    stride_qz, stride_qm, stride_qh,
    stride_kz, stride_km, stride_kh,
    stride_vz, stride_vm, stride_vh,
    stride_oz, stride_om, stride_oh,
    stride_block_table_seq,
    stride_k_block, stride_k_head, stride_k_bs, stride_k_dim,
    stride_v_block, stride_v_head, stride_v_bs, stride_v_dim,
    # === 编译期常量 ===
    total_tasks, 
    NUM_HEADS: tl.constexpr, 
    NUM_KV_HEADS: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, # 通常 16
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # =======================================================
    # 1. 抢占任务与全局信息解算
    # =======================================================
    task_id = tl.atomic_add(tile_counter_ptr, 1)

    while task_id < total_tasks:  #  修改成  for _ in range(total_tasks): if task_id < total_tasks:以前遇到triton无法处理while循环的问题，但能通过for循环解决
        #这里是否真的能预算出total_tasks，maybe可以
        head_idx = task_id % NUM_HEADS
        tile_m_and_seq = task_id // NUM_HEADS
        
        seq_idx = tl.load(tile2seq_ptr + tile_m_and_seq)#位于第几个batch
        seq_tile_start = tl.load(cu_tiles + seq_idx)
        tile_m_idx = tile_m_and_seq - seq_tile_start# 公式：全局 Tile 编号 - 当前句子第一块 Tile 的全局编号 = 当前句子内部的相对 Tile 编号！
        
        # 长度换算 (核心)
        seq_tok_start = tl.load(cu_seqlens + seq_idx)
        new_len = tl.load(cu_seqlens + seq_idx + 1) - seq_tok_start # 新词长度
        total_len = tl.load(context_lens + seq_idx)                 # 总长度 (历史 + 新词)
        prefix_len = total_len - new_len                            # 历史前缀长度
        
        kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)

        # 当前 Q Block 的局部 Token 偏移
        offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_K)

        # =======================================================
        # 2. Q 向量 TMA 读取 (仅针对当前的新词)
        # =======================================================
        q_ptr = tl.make_block_ptr(
            base=Q + head_idx * stride_qh,
            shape=(seq_tok_start + new_len, BLOCK_K),
            strides=(stride_qm, 1),
            offsets=(seq_tok_start + tile_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        q = tl.load(q_ptr, boundary_check=(0, 1))

        # =======================================================
        # 3. Softmax 状态初始化
        # =======================================================
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        qk_scale_adj = sm_scale * 1.44269504 # log2(e) 优化计算

        # =======================================================
        # 4. Stage A: 吸纳历史 Prefix (读 kv_cache)
        # =======================================================
        for start_n in range(0, prefix_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            # Paged 查表获取物理块
            logical_blocks = offs_n // BLOCK_SIZE
            offsets_in_block = offs_n % BLOCK_SIZE
            physical_blocks = tl.load(block_tables + seq_idx * stride_block_table_seq + logical_blocks, 
                                      mask=offs_n < prefix_len, other=0)
            
            # 128-bit 异步指针算术
            k_ptrs = K_cache + physical_blocks[:, None] * stride_k_block + kv_head_idx * stride_k_head + offsets_in_block[:, None] * stride_k_bs + offs_d[None, :] * stride_k_dim
            v_ptrs = V_cache + physical_blocks[:, None] * stride_v_block + kv_head_idx * stride_v_head + offsets_in_block[:, None] * stride_v_bs + offs_d[None, :] * stride_v_dim
            
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < prefix_len), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < prefix_len), other=0.0)
            
            # Attention 计算 (历史对新词全部可见，无 Causal Mask)
            qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale_adj
            qk = tl.where(offs_n[None, :] < prefix_len, qk, -1.0e20)
            
            # Base-2 Online Softmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_ij

        # =======================================================
        # 5. Stage B: 处理新词 (读 QKV 原张量 + 写落盘)
        # =======================================================
        for start_n in range(prefix_len, total_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            local_n = offs_n - prefix_len # 在新词张量中的偏移
            mask_new = local_n < new_len
            
            # [读]：直接从输入的连续张量中 128-bit 读取
            k_in_ptrs = K + kv_head_idx * stride_kh + (seq_tok_start + local_n)[:, None] * stride_km + offs_d[None, :] * 1
            v_in_ptrs = V + kv_head_idx * stride_vh + (seq_tok_start + local_n)[:, None] * stride_vm + offs_d[None, :] * 1
            
            k_new = tl.load(k_in_ptrs, mask=mask_new[:, None], other=0.0)
            v_new = tl.load(v_in_ptrs, mask=mask_new[:, None], other=0.0)
            
            # [写]：Fused Cache (极致去重优化)
            # 只有头组的第一个 Head，并且只有处理该 Sequence 第一个 Q Block 时，才负责全量写入！
            if (head_idx % (NUM_HEADS // NUM_KV_HEADS) == 0) and (tile_m_idx == 0):
                slots = tl.load(slot_mapping + seq_tok_start + local_n, mask=mask_new, other=0)
                logical_blocks = slots // BLOCK_SIZE
                offsets_in_block = slots % BLOCK_SIZE
                
                k_write_ptrs = K_cache + logical_blocks[:, None] * stride_k_block + kv_head_idx * stride_k_head + offsets_in_block[:, None] * stride_k_bs + offs_d[None, :] * stride_k_dim
                v_write_ptrs = V_cache + logical_blocks[:, None] * stride_v_block + kv_head_idx * stride_v_head + offsets_in_block[:, None] * stride_v_bs + offs_d[None, :] * stride_v_dim
                
                tl.store(k_write_ptrs, k_new, mask=mask_new[:, None])
                tl.store(v_write_ptrs, v_new, mask=mask_new[:, None])
                
            # Attention 计算
            qk = tl.dot(q, tl.trans(k_new)).to(tl.float32) * qk_scale_adj
            
            # Causal Mask (新词相对前缀进行绝对坐标掩码)
            if IS_CAUSAL:
                causal_mask = (prefix_len + offs_m)[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask & mask_new[None, :], qk, -1.0e20)
            else:
                qk = tl.where(mask_new[None, :], qk, -1.0e20)
                
            # 无缝衔接 Online Softmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v_new.dtype), v_new)
            m_i = m_ij

        # =======================================================
        # 6. 收尾：输出结果 TMA 写回
        # =======================================================
        acc = acc / l_i[:, None]
        
        o_ptr = tl.make_block_ptr(
            base=Out + head_idx * stride_oh,
            shape=(seq_tok_start + new_len, BLOCK_K),
            strides=(stride_om, 1),
            offsets=(seq_tok_start + tile_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        tl.store(o_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))

        # 拉取下个任务
        task_id = tl.atomic_add(tile_counter_ptr, 1)


@triton.jit
def _paged_attn_decode_kernel(
    Q, K_new, V_new, Out,
    # === Paged KV 物理池与寻址表 ===
    K_cache, V_cache,
    block_tables, slot_mapping,
    # === 序列长度控制 ===
    context_lens,         # [num_seqs] 每个序列当前的总长度 (包含新词)
    sm_scale,
    # === 张量 Stride ===
    stride_q_seq, stride_q_head, stride_q_dim,
    stride_knew_seq, stride_knew_head, stride_knew_dim,
    stride_vnew_seq, stride_vnew_head, stride_vnew_dim,
    stride_o_seq, stride_o_head, stride_o_dim,
    stride_block_table_seq,
    stride_k_block, stride_k_head, stride_k_bs, stride_k_dim,
    stride_v_block, stride_v_head, stride_v_bs, stride_v_dim,
    # === 编译期常量 ===
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, # 物理页大小 (如 16)
    BLOCK_D: tl.constexpr,    # Head 维度 (如 64 或 128)
):
    # =======================================================
    # 1. Decode 的 Grid 映射 (非 Persistent，按 Batch 和 Head 展开)
    # =======================================================
    # Grid = (num_seqs, num_heads)
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)
    ctx_len = tl.load(context_lens + seq_idx)
    prefix_len = ctx_len - 1     # 历史长度（新 token 之前的 token 数）

    offs_d = tl.arange(0, BLOCK_D)

    # =======================================================
    # 2. KV 落盘：把当前新 token 的 K/V 写进 paged cache（供后续 decode step 读取）
    # =======================================================
    # 去重：只有每个 KV 组的第一个 Q Head 负责写一次。
    # 注意：这里写的是「新 token 的 slot」，而第 4 步只读「历史」（mask 掉新 token 位置），
    # 第 5 步读新 token 走的是 input（K_new/V_new），所以写 cache 与读 cache 互不重叠，
    # 不存在跨 CTA 竞争。
    if head_idx % (NUM_HEADS // NUM_KV_HEADS) == 0:
        k_new = tl.load(K_new + seq_idx * stride_knew_seq + kv_head_idx * stride_knew_head + offs_d * stride_knew_dim)
        v_new = tl.load(V_new + seq_idx * stride_vnew_seq + kv_head_idx * stride_vnew_head + offs_d * stride_vnew_dim)

        slot = tl.load(slot_mapping + seq_idx)
        block_idx = slot // BLOCK_SIZE
        block_offset = slot % BLOCK_SIZE

        k_cache_ptr = K_cache + block_idx * stride_k_block + kv_head_idx * stride_k_head + block_offset * stride_k_bs + offs_d * stride_k_dim
        v_cache_ptr = V_cache + block_idx * stride_v_block + kv_head_idx * stride_v_head + block_offset * stride_v_bs + offs_d * stride_v_dim
        tl.store(k_cache_ptr, k_new)
        tl.store(v_cache_ptr, v_new)

    # =======================================================
    # 3. 读取 Q 与新 token 的 K/V（从 input 读，避免与上面的落盘竞争）
    # =======================================================
    # Decode 时 Q 长度恒为 1，读成 [1, BLOCK_D] 二维张量以适配 tl.dot
    q = tl.load(Q + seq_idx * stride_q_seq + head_idx * stride_q_head + offs_d * stride_q_dim)[None, :]
    k_new = tl.load(K_new + seq_idx * stride_knew_seq + kv_head_idx * stride_knew_head + offs_d * stride_knew_dim)  # [BLOCK_D]
    v_new = tl.load(V_new + seq_idx * stride_vnew_seq + kv_head_idx * stride_vnew_head + offs_d * stride_vnew_dim)  # [BLOCK_D]

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([1, BLOCK_D], dtype=tl.float32)

    qk_scale_adj = sm_scale * 1.44269504

    # =======================================================
    # 4. 核心循环：按 BLOCK_SIZE 遍历「历史」Paged Cache（不含新 token）
    # =======================================================
    for start_n in range(0, prefix_len, BLOCK_SIZE):
        offs_n = tl.arange(0, BLOCK_SIZE)
        global_n = start_n + offs_n

        # 1. 查表：当前循环对应第几个逻辑块？物理块是谁？
        logical_block_idx = start_n // BLOCK_SIZE
        physical_block_idx = tl.load(block_tables + seq_idx * stride_block_table_seq + logical_block_idx)

        # 2. 构建二维指针 [BLOCK_SIZE, BLOCK_D]
        k_ptrs = K_cache + physical_block_idx * stride_k_block + kv_head_idx * stride_k_head + offs_n[:, None] * stride_k_bs + offs_d[None, :] * stride_k_dim
        v_ptrs = V_cache + physical_block_idx * stride_v_block + kv_head_idx * stride_v_head + offs_n[:, None] * stride_v_bs + offs_d[None, :] * stride_v_dim

        # 3. 边界遮罩：只保留历史位置，mask 掉新 token 位置（与第 2 步落盘不重叠）
        mask = global_n < prefix_len

        # 4. 加载 K 和 V
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)

        # 5. 计算注意力得分 Q * K^T (形状 [1, BLOCK_D] x [BLOCK_D, BLOCK_SIZE] -> [1, BLOCK_SIZE])
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale_adj
        qk = tl.where(mask[None, :], qk, -1.0e20)

        # 6. Online Softmax 极速更新
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        p = tl.exp2(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp2(m_i - m_i_new)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_i_new

    # =======================================================
    # 5. 处理新 token：从 input 读（新 token 一定可见，无需 mask）
    # =======================================================
    qk_new = tl.dot(q, tl.reshape(k_new, [BLOCK_D, 1])).to(tl.float32) * qk_scale_adj  # [1, 1]
    m_ij = tl.max(qk_new, 1)
    m_i_new = tl.maximum(m_i, m_ij)

    p_new = tl.exp2(qk_new - m_i_new[:, None])   # [1, 1]
    l_ij = tl.sum(p_new, 1)

    alpha = tl.exp2(m_i - m_i_new)
    l_i = l_i * alpha + l_ij

    acc = acc * alpha[:, None]
    acc += tl.reshape(v_new, [1, BLOCK_D]) * p_new.to(v_new.dtype)

    m_i = m_i_new

    # =======================================================
    # 6. 输出写回
    # =======================================================
    acc = acc / l_i[:, None]

    # 降维回 [BLOCK_D] 一维张量并写入（acc 是 [1, BLOCK_D]，用 reshape 压平，
    # 不能用 acc[0] 常量索引 —— Triton 3.x 不支持）
    out_ptr = Out + seq_idx * stride_o_seq + head_idx * stride_o_head + offs_d * stride_o_dim
    tl.store(out_ptr, tl.reshape(acc, [BLOCK_D]).to(Out.type.element_ty))

tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K' : BK}, num_stages=s, num_warps=w) \
    for BM in [128]\
    for BN in [128, 256]\
    for BK in [64, 128]\
    for s in ([3, 4])\
    for w in [4, 8]\
]

@triton.autotune(
    tma_configs,
    key=['group_size'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles
#=====================================================================
#中间工具函数
#=====================================================================
DEVICE = triton.runtime.driver.active.get_active_torch_device()
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

def group_gemm_tma_fn(group_A, group_B):

    assert supports_tma()

    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[1]
        M, K = A.shape
        N, K = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    # we use a fixed number of CTA, and it's auto-tunable

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_tma_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
                                    FP8=torch.float8_e4m3fn == group_A[0].dtype, NUM_SM=num_sms())
    return group_C

# =====================================================================
# 1. FlashAttention 内核封装 (纯推理，无状态，不写 backward)
# =====================================================================

class FlashAttentionWrapper(Attention):
    """
    负责提取 vLLM 元数据，进行 Prefill / Decode 分流，
    并将张量与 Paged KV 物理池适配送入底层的 Triton 核函数。

    关键设计（方案 A / "伪装" vLLM 原生 Attention）：
      继承 vllm.model_executor.layers.attention.Attention，让 vLLM 把本层
      识别为注意力层，从而自动完成三件事：
        1) 把本层注册进 static_forward_context（调度器据此分配 KV cache 组）；
        2) 用 FlashAttention 后端构建 FlashAttentionMetadata（含
           block_table / slot_mapping / query_start_loc / seq_lens /
           num_prefill_tokens / num_decode_tokens 等内核需要的字段）；
        3) 分配并 bind 物理 KV cache（self.kv_cache）。
      但 forward() 里我们完全不走它的底层实现，而是跑自己的 Triton 核。
    """
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 layer_name: str = "layers.0.self_attn", warp_specialize: bool = True):
        # 调用 vLLM Attention 的 __init__（注册本层 + 选择后端 + 建立 kv_cache 占位）。
        # 注意 layer_name 必须含一个整数下标（vLLM 的 extract_layer_index 依赖它）。
        super().__init__(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / math.sqrt(head_dim),
            num_kv_heads=num_kv_heads,
            prefix=layer_name,
        )
        self.head_dim = head_dim
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        assert head_dim in {32, 64, 128, 256}, f"TMA 模式下 Head Dim ({head_dim}) 必须是 32/64/128/256"
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = head_dim
        if warp_specialize:
            self.extra_kwargs = {"num_warps": 8, "num_stages": 4}
        else:
            self.extra_kwargs = {"num_warps": 4, "num_stages": 2}

    def _split_kv_cache(self, kv_cache):
        """把 vLLM 的物理 KV 池拆成内核要的 K_cache / V_cache。

        内核要求的布局是 [num_blocks, num_kv_heads, block_size, head_dim]。
        兼容三种来源：
          1) vLLM 0.29 LBNHC 4D：[num_blocks, num_kv_heads, block_size, 2*head_dim]（K/V 拼在最后一维）
          2) 旧版 V1 5D：[num_blocks, num_kv_heads, 2, block_size, head_dim]
          3) 直接传入 (k_cache, v_cache) 元组
        """
        if kv_cache is None:
            raise ValueError(f"[{self.layer_name}] kv_cache 不能为 None")
        if kv_cache.dim() == 4:
            # 直接在最后一个维度（dim=-1）上切分为 k 和 v
            # 返回的直接就是形状为 [B, H, N, D] 的视图，零拷贝，步长天然正确
            k_cache, v_cache = kv_cache.split(self.head_dim, dim=-1)
            block_size = kv_cache.shape[2]
        elif kv_cache.dim() == 5:
            # [num_blocks, num_kv_heads, 2, block_size, head_dim]
            k_cache = kv_cache[:, :, 0, :, :]
            v_cache = kv_cache[:, :, 1, :, :]
            block_size = kv_cache.shape[3]
        elif isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
            k_cache, v_cache = kv_cache
            block_size = k_cache.shape[2]
        else:
            raise ValueError(
                f"[{self.layer_name}] 不支持的 kv_cache 布局: "
                f"{tuple(kv_cache.shape) if hasattr(kv_cache, 'shape') else type(kv_cache)}"
            )
        return k_cache, v_cache, block_size

    def forward(
        self,
        query: torch.Tensor,          # [Total_Tokens, NUM_HEADS, HEAD_DIM]
        key: torch.Tensor,            # [Total_Tokens, NUM_KV_HEADS, HEAD_DIM]
        value: torch.Tensor,          # [Total_Tokens, NUM_KV_HEADS, HEAD_DIM]
        output_shape: torch.Size | None = None,
        output_dtype: torch.dtype | None = None,
    ):
        # 与 vLLM Attention.forward 保持一致的调用签名（query/key/value 位置参数），
        # 但内部完全绕过 vLLM 后端实现，改跑自写 Triton 核。
        q, k, v = query, key, value
        total_tokens, num_heads, head_dim = q.shape
        _, num_kv_heads, _ = k.shape

        # 获取 vLLM 调度元数据（key 就是本层的 layer_name）
        if not is_forward_context_available():
            raise RuntimeError(f"[{self.layer_name}] 必须在 vLLM forward_context 内运行！")

        fc = get_forward_context()
        attn_metadata = fc.attn_metadata
        # profile_run（determine_available_memory 的 dummy run）以 skip_attn=True 执行，
        # 此时不构建 attention metadata、也尚未 bind KV cache，直接透传占位。
        if attn_metadata is None:
            return query
        meta = attn_metadata[self.layer_name]

        # 混合批次的边界信息
        # 注意：vLLM 0.29 单卡下 FlashAttentionMetadata.num_prefill_tokens / num_decode_tokens
        # 恒为 0（它们只用于 DCP 解码上下文并行），必须自己用 query_start_loc 计算边界。
        num_actual_tokens = int(meta.num_actual_tokens)
        max_query_len = int(meta.max_query_len)
        if max_query_len <= 1:
            # 纯 decode
            num_prefill_tokens = 0
            num_decode_tokens = num_actual_tokens
            num_prefill_reqs = 0
        else:
            # 含 prefill（vLLM V1 约定：batch 里 prefill 在前、decode 在后）
            query_start_loc = meta.query_start_loc  # [0, 15, 25, 26, 27]
            query_lens = query_start_loc[1:] - query_start_loc[:-1]# 结果 query_lens = [15, 10, 1, 1]
            num_decode_reqs = int((query_lens == 1).sum().item())# 2. 找 Decode：统计长度等于 1 的请求数量-->2
            num_decode_tokens = num_decode_reqs# 3. Decode Token 数
            num_prefill_tokens = num_actual_tokens - num_decode_tokens# 4. Prefill Token 数
            num_prefill_reqs = query_start_loc.numel() - 1 - num_decode_reqs#5. Prefill 请求数

        # 阶段识别打印（仅当 SELF_LLM_DEBUG=1 时，用于观察推理阶段）
        if os.environ.get("SELF_LLM_DEBUG", "0") == "1":
            if num_prefill_tokens > 0:
                qlens = meta.query_start_loc[1:num_prefill_reqs + 1] - meta.query_start_loc[:num_prefill_reqs]
                clens = meta.seq_lens[:num_prefill_reqs]
                plens = clens - qlens
                tag = "prefix-prefill(有历史)" if bool((plens > 0).any().item()) else "纯prefill(无历史)"
                print(f"    [阶段] PREFILL: {num_prefill_tokens} token / {num_prefill_reqs} 请求 | "
                      f"query_lens={qlens.tolist()} context_lens={clens.tolist()} → {tag}")
            if num_decode_tokens > 0:
                print(f"    [阶段] DECODE: {num_decode_tokens} token")

        # 统一预分配完整输出张量与缩放因子
        out = torch.empty_like(q)
        sm_scale = 1.0 / (head_dim ** 0.5)
        causal = True  # 本模型是 decoder-only，prefill 做因果注意力

        # 统一拆分物理 KV 池 -> K_cache / V_cache（形状: [num_blocks, num_kv_heads, block_size, head_dim]）
        K_cache, V_cache, block_size = self._split_kv_cache(self.kv_cache)

        # =========================================================
        # 1. 一次性处理所有 Prefill (Ragged Persistent Attention)
        # =========================================================
        if num_prefill_tokens > 0:
            # num_prefill_reqs 已在上方根据 query_start_loc 计算

            # 物理内存切片：精准切出 Prefill 专属的部分
            q_prefill = q[:num_prefill_tokens]
            k_prefill = k[:num_prefill_tokens]
            v_prefill = v[:num_prefill_tokens]
            out_prefill = out[:num_prefill_tokens]
            slot_mapping_prefill = meta.slot_mapping[:num_prefill_tokens]

            # Prefill 动态元数据
            cu_seqlens = meta.query_start_loc[:num_prefill_reqs + 1]  # 累计偏移
            seq_lens_prefill = meta.seq_lens[:num_prefill_reqs]        # 每条序列总长(含历史)
            block_tables_prefill = meta.block_table[:num_prefill_reqs] # 分页块表

            # --- 构建 Persistent Task 映射表 ---
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            
            # Triton Block 配置 (可根据需求调整或从外部传入)
            block_m = getattr(self, "BLOCK_M", 64)
            block_n = getattr(self, "BLOCK_N", 64)
            block_k = head_dim
            
            tiles_per_seq = (seqlens + block_m - 1) // block_m
            cu_tiles = torch.zeros_like(cu_seqlens)
            cu_tiles[1:] = torch.cumsum(tiles_per_seq, dim=0)

            total_tasks = cu_tiles[-1].item() * num_heads
            num_seqs = cu_seqlens.numel() - 1

            seq_ids = torch.arange(num_seqs, device=q.device, dtype=torch.int32)
            tile2seq_ptr = torch.repeat_interleave(seq_ids, tiles_per_seq).contiguous()
            tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)

            # 动态 Grid 计算
            device_id = q.device.index if q.device.index is not None else 0
            num_sms = torch.cuda.get_device_properties(device_id).multi_processor_count
            grid_prefill = (min(num_sms, total_tasks), )

            # 拉起 _ragged_attn_fwd_kernel_persistent
            _ragged_attn_fwd_kernel_persistent[grid_prefill](
                q_prefill, k_prefill, v_prefill, out_prefill,
                K_cache, V_cache,
                block_tables_prefill, slot_mapping_prefill,
                cu_tiles, cu_seqlens, seq_lens_prefill,
                tile2seq_ptr, tile_counter,
                sm_scale,
                # 张量 Stride (Q, K, V, Out)
                q_prefill.stride(0), q_prefill.stride(1), q_prefill.stride(2),
                k_prefill.stride(0), k_prefill.stride(1), k_prefill.stride(2),
                v_prefill.stride(0), v_prefill.stride(1), v_prefill.stride(2),
                out_prefill.stride(0), out_prefill.stride(1), out_prefill.stride(2),
                # Block Table Stride
                block_tables_prefill.stride(0),
                # K/V Cache Stride: [block, head, bs, dim]
                K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
                V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
                # 编译期常量
                total_tasks=total_tasks,
                NUM_HEADS=num_heads,
                NUM_KV_HEADS=num_kv_heads,
                BLOCK_SIZE=block_size,
                IS_CAUSAL=causal,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
            )

        # =========================================================
        # 2. 一次性处理所有 Decode (基于 Paged KV 查表的向量乘法)
        # =========================================================
        if num_decode_tokens > 0:
            # num_prefill_reqs 已在上方根据 query_start_loc 计算

            # 物理内存切片：精准切出 Decode 专属的部分
            q_decode = q[num_prefill_tokens:]
            k_decode = k[num_prefill_tokens:]
            v_decode = v[num_prefill_tokens:]
            out_decode = out[num_prefill_tokens:]
            slot_mapping_decode = meta.slot_mapping[num_prefill_tokens:]

            # 提取 Decode 所需的专属查表数据（跳过 prefill 的请求记录）
            block_tables_decode = meta.block_table[num_prefill_reqs:]
            context_lens_decode = meta.seq_lens[num_prefill_reqs:]

            # Grid 形状：(Decode Token 总数，Query 头数)
            grid_decode = (num_decode_tokens, num_heads)

            # 拉起 _paged_attn_decode_kernel（KV 落盘 + 注意力都在核内完成）
            _paged_attn_decode_kernel[grid_decode](
                q_decode, k_decode, v_decode, out_decode,
                K_cache, V_cache,
                block_tables_decode, slot_mapping_decode,
                context_lens_decode,
                sm_scale,
                # 张量 Stride (Q, K_new, V_new, Out)
                q_decode.stride(0), q_decode.stride(1), q_decode.stride(2),
                k_decode.stride(0), k_decode.stride(1), k_decode.stride(2),
                v_decode.stride(0), v_decode.stride(1), v_decode.stride(2),
                out_decode.stride(0), out_decode.stride(1), out_decode.stride(2),
                # Block Table Stride
                block_tables_decode.stride(0),
                # K/V Cache Stride: [block, head, bs, dim]
                K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
                V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
                # 编译期常量
                NUM_HEADS=num_heads,
                NUM_KV_HEADS=num_kv_heads,
                BLOCK_SIZE=block_size,
                BLOCK_D=head_dim,
            )

        # 完美组装：Prefill 和 Decode 的计算结果已经在各自的切片视图下直接落入了完整的 out 内存中
        return out


# =====================================================================
# 2. MoE 专家层 (含 Router 打分、Permute/Unpermute 和 Grouped GEMM 调用)
# =====================================================================
class MoEInferenceLayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ffn = d_ffn

        # Router 路由权重  —— 全链路统一 fp16
        self.router = nn.Linear(d_model, num_experts, bias=False, dtype=torch.float16)
        
        # 专家权重矩阵：E 个专家的 Up 和 Down 投影矩阵
        # ==================== 核心修改点 ====================
        # 为了完美匹配 Triton TMA Grouped GEMM 计算规范 A @ B^T (要求 A.shape[1] == B.shape[1])
        # W1 (Up-proj):   输入 A 为 [M, d_model]，故要求 B 的形状为 [d_ffn, d_model]
        # W2 (Down-proj): 输入 A 为 [M, d_ffn]，  故要求 B 的形状为 [d_model, d_ffn]
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ffn, d_model, dtype=torch.float16))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ffn, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_model]
        N, d_model = x.shape

        # =====================================================================
        # 1. Router 打分与 Top-K 选择 (保持不变，全部在 GPU 异步运行)
        # =====================================================================
        router_logits = self.router(x)  # [N, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1) # [N, top_k]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # 归一化 [N, top_k]

        # =====================================================================
        # 2. Permute: 根据选中专家对 Token 展平并按 Expert ID 排序 (保持不变)
        # =====================================================================
        flat_topk_indices = topk_indices.view(-1)  # [N * top_k]
        flat_x = x.repeat_interleave(self.top_k, dim=0) # [N * top_k, d_model]

        # 排序：将发往相同专家的 Token 排在连续的内存区域
        sorted_expert_ids, permute_indices = torch.sort(flat_topk_indices)
        permuted_x = flat_x[permute_indices] # [N * top_k, d_model]

        # 计算每个专家分到的 Token 数量
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)

        # =====================================================================
        # 3. 调用 Triton Grouped GEMM 执行专家计算 (★ 核心重构区域 ★)
        # =====================================================================
        
        # 【修改点 1】：彻底废弃 torch.split 和 tolist()
        # 不再把连续的显存切碎成 Python list。而是提前分配好一整块大的连续显存作为输出，
        # 直接让 Triton Kernel 把结果写进对应的连续偏移位置。这省去了最后的 torch.cat。
        total_tokens = permuted_x.shape[0]
        device = permuted_x.device
        dtype = permuted_x.dtype
        element_size = permuted_x.element_size() # FP16 占 2 字节

        permuted_C_w1 = torch.empty((total_tokens, self.d_ffn), dtype=dtype, device=device)
        permuted_out = torch.empty((total_tokens, d_model), dtype=dtype, device=device)

        # 【修改点 2】：利用 GPU 的前缀和 (cumsum) 算好指针偏移量
        # expert_offsets 记录了每个专家在连续大矩阵中的“行号”起点。
        # 这里全都是 Tensor 运算，CPU 不需要等待读取具体数值！
        zero = torch.zeros(1, dtype=torch.long, device=device)
        expert_offsets = torch.cat([zero, torch.cumsum(expert_counts, dim=0)])[:-1]

        # --- 内部闭包函数：在显存里直接用 Tensor 拼装好 TMA 算子需要的元数据 ---
        def build_tma_args(A, B_3d, C):
            # 算出所有矩阵在 VRAM 里的真实物理起始地址指针 (Data Pointers)
            d_a = A.data_ptr() + expert_offsets * (A.stride(0) * element_size)
            d_b = B_3d.data_ptr() + torch.arange(self.num_experts, device=device) * (B_3d.stride(0) * element_size)
            d_c = C.data_ptr() + expert_offsets * (C.stride(0) * element_size)

            # 拼装 Group GEMM 的尺寸数组 [M0,N0,K0, M1,N1,K1...]
            N_dim, K_dim = B_3d.shape[1], B_3d.shape[2]
            N_tensor = torch.full((self.num_experts,), N_dim, dtype=torch.int32, device=device)
            K_tensor = torch.full((self.num_experts,), K_dim, dtype=torch.int32, device=device)
            # 注意：expert_counts 包含 0，传进内核后 Triton 遇到 M=0 会自动跳过计算
            d_size = torch.stack([expert_counts.to(torch.int32), N_tensor, K_tensor], dim=1).view(-1)

            # 拼装步长
            d_lds = torch.tensor([A.stride(0), B_3d.stride(1), C.stride(0)], dtype=torch.int32, device=device).repeat(self.num_experts)
            return d_a, d_b, d_c, d_size, d_lds

        # 【修改点 3】：按原逻辑设置分配器，准备发射内核
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)

        sm_count = num_sms()
        grid = lambda META: (sm_count, )
        is_fp8 = (dtype == torch.float8_e4m3fn)

        # ---------------------------------------------------------------------
        # Step A: 第一次 GEMM (W1)
        # ---------------------------------------------------------------------
        d_a1, d_b1, d_c1, d_size1, d_lds1 = build_tma_args(permuted_x, self.w1, permuted_C_w1)
        grouped_matmul_tma_kernel[grid](
            d_a1, d_b1, d_c1, d_size1, d_lds1, self.num_experts,
            FP8=is_fp8, NUM_SM=sm_count
        )

        # ---------------------------------------------------------------------
        # Step B: 激活函数 SiLU 
        # 【修改点 4】：由于 permuted_C_w1 是一整块矩阵，这里不用写 for 循环了。
        # GPU 直接一波把所有的计算完，性能拉满！
        # ---------------------------------------------------------------------
        active_A_w2 = F.silu(permuted_C_w1)

        # ---------------------------------------------------------------------
        # Step C: 第二次 GEMM (W2)
        # ---------------------------------------------------------------------
        d_a2, d_b2, d_c2, d_size2, d_lds2 = build_tma_args(active_A_w2, self.w2, permuted_out)
        grouped_matmul_tma_kernel[grid](
            d_a2, d_b2, d_c2, d_size2, d_lds2, self.num_experts,
            FP8=is_fp8, NUM_SM=sm_count
        )
        
        # 此时的 permuted_out 已经是所有专家计算完毕且组装好的完整 [N * top_k, d_model] 矩阵了

        # =====================================================================
        # 4. Unpermute: 还原 Token 原始顺序并加权求和 (保持不变)
        # =====================================================================
        # 反向映射索引
        unpermute_indices = torch.empty_like(permute_indices)
        unpermute_indices[permute_indices] = torch.arange(permute_indices.size(0), device=x.device)
        
        restored_out = permuted_out[unpermute_indices] # [N * top_k, d_model]
        restored_out = restored_out.view(N, self.top_k, d_model)

        # 乘以 Top-K 路由权重合并输出
        final_out = (restored_out * topk_weights.unsqueeze(-1)).sum(dim=1) # [N, d_model]
        return final_out

# =====================================================================
# 3. GroupTransformer 总控制类 (完整的 Transformer Block)
# =====================================================================
class GroupTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,        # = config.json.hidden_size
        num_heads: int = 8,        # = config.json.num_attention_heads
        num_kv_heads: int = 2,     # = config.json.num_key_value_heads  (GQA 支持：8 Q heads 共享 2 KV heads)
        head_dim: int = 64,        # = config.json.head_dim
        num_experts: int = 8,      # = config.json.num_experts
        top_k: int = 2,            # = config.json.top_k
        d_ffn: int = 2048,         # = config.json.d_ffn
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 归一化层 (Pre-LN)  —— 全链路统一 fp16
        self.input_norm = nn.RMSNorm(d_model, dtype=torch.float16)
        self.post_attn_norm = nn.RMSNorm(d_model, dtype=torch.float16)

        # Attention 线性投影层 (标准 GEMM)  —— 全链路统一 fp16
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=torch.float16)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=torch.float16)

        # MoE 专家机制层
        self.moe_layer = MoEInferenceLayer(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            d_ffn=d_ffn
        )
        self.attention_layer = FlashAttentionWrapper(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_name="layers.0.self_attn",  # 必须含整数下标，vLLM 的 extract_layer_index 依赖
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        vLLM 版前向：hidden_states → Attention 子层 + MoE 子层 → hidden_states。

        hidden_states: [num_tokens, d_model] - 多个变长序列打平拼接后的输入（vLLM Ragged）
        positions:     [num_tokens] - 每个 token 的绝对位置（暂未使用，将来 RoPE 用）
        注意：KV cache 与 attention metadata 由 attention_layer 内部从
        self.kv_cache / forward_context.attn_metadata 获取，不再显式传参。
        """
        N, d_model = hidden_states.shape

        # ==========================================
        # 1. Attention 子层 (+ 残差连接)
        # ==========================================
        residual = hidden_states
        x_norm = self.input_norm(hidden_states)

        # (1) QKV 线性投影
        q = self.q_proj(x_norm).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_norm).view(N, self.num_kv_heads, self.head_dim)
        #self.q_proj(x_norm)完成X*WQ，.view(N, self.num_heads, self.head_dim)拆分三维

        attn_out = self.attention_layer(q, k, v)

        # (3) 输出拼接与投影
        attn_out = attn_out.view(N, self.num_heads * self.head_dim)
        attn_out = self.out_proj(attn_out)
        #多头信息融合：FlashAttention 计算完后，各个 Head 的注意力特征是独立分开的。乘以 $W_O$ 矩阵可以把所有 Head 抽取到的特征进行混合与信息融合。

        # (4) 残差相加
        hidden_states = residual + attn_out

        # ==========================================
        # 2. MoE 子层 (+ 残差连接)
        # ==========================================
        residual = hidden_states
        x_norm = self.post_attn_norm(hidden_states)

        # (1) MoE 专家计算 (Router -> Permute -> Grouped GEMM -> Unpermute)
        moe_out = self.moe_layer(x_norm)

        # (2) 残差相加
        hidden_states = residual + moe_out

        return hidden_states


# =====================================================================
# 模型接口：SelfLLM（vLLM 0.29.0 / V1 兼容）
# =====================================================================
# 三层设计：
#   第一层 embedding ：self.embed_tokens  —— token id → d_model 维向量
#   第二层 transform ：self.transformer   —— 占位 None（TODO：将来替换成 GroupTransformer）
#   第三层 预测层    ：self.lm_head       —— hidden → logits
# BPE 分词由 vLLM 主机侧完成，这里直接接收 token id，只需做 embedding。


class SelfLLM(nn.Module):
    """vLLM 0.29.0 (V1) 兼容模型类：embedding → GroupTransformer（占位）→ lm_head。

    vLLM 调用约定：
      - SelfLLM(vllm_config=..., prefix=...)           keyword-only
      - embed_input_ids(input_ids)
      - forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None)
      - compute_logits(hidden_states)
      - load_weights(weights) -> set[str]
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.vocab_size = getattr(hf_config, "vocab_size", 151936)
        self.d_model = getattr(hf_config, "hidden_size", 512)
        self.num_hidden_layers = getattr(hf_config, "num_hidden_layers", 1)
        self.num_attention_heads = getattr(hf_config, "num_attention_heads", 8)
        self.num_key_value_heads = getattr(hf_config, "num_key_value_heads", 2)
        self.head_dim = getattr(hf_config, "head_dim", 64)
        self.num_experts = getattr(hf_config, "num_experts", 8)
        self.top_k = getattr(hf_config, "top_k", 2)
        self.d_ffn = getattr(hf_config, "d_ffn", 2048)

        # 第一层：embedding（token id → 向量）  —— 全链路统一 fp16，与 GroupTransformer 对齐
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model, dtype=torch.float16)

        # 第二层：transform —— GroupTransformer（attention + MoE）
        # attention 已接入 vLLM：FlashAttentionWrapper 继承 vLLM 的 Attention 层做 KV cache
        # 管理 + metadata 构建，forward 里跑自写 Triton 核（prefill 用 ragged persistent、
        # decode 用 paged 只读核）。KV 落盘统一由 prefill 核 / decode 前的 torch scatter 完成。
        self.transformer = GroupTransformer(
            d_model=self.d_model,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            d_ffn=self.d_ffn,
        )

        # 第三层：预测层  —— 全链路统一 fp16
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=torch.float16)

        self._forward_calls = 0

    # ---- embedding 端接口（vLLM 强制）----
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._forward_calls += 1
        if self._forward_calls <= 10:
            print(f"[SELF-LLM.py::SelfLLM.forward #{self._forward_calls}] "
                  f"input_ids={tuple(input_ids.shape) if input_ids is not None else None} "
                  f"positions={tuple(positions.shape)}")

        '''
        if intermediate_tensors is not None:
            # 场景 A：我是 GPU 1，接收前半截传来的半成品
            hidden_states = intermediate_tensors["hidden_states"]
        else:
            # 场景 B：我是 GPU 0，接收最原始的文本 ID（或图像向量）
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        '''

        # 第一层：embedding
        if inputs_embeds is not None:#不允许同时存在inputs_embeds和input_ids
            hidden = inputs_embeds
        else:
            hidden = self.embed_tokens(input_ids)  # [*token_shape, d_model]

        # 统一展平成 [num_tokens, d_model]（vLLM Ragged 约定：多序列 token 打平）
        # 兼容 vLLM 直接传 [num_tokens] 或 [batch, seq] 两种 input_ids 形状
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, self.d_model)

        # 第二层：transform（GroupTransformer：attention + MoE）
        if self.transformer is not None:
            hidden = self.transformer(hidden, positions)

        return hidden

    # ---- 预测端接口（vLLM 强制）----
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    # ---- 权重加载接口（vLLM 强制，延迟 import 避免纯 torch 侧依赖 vllm）----
    def load_weights(self, weights):
        """按参数名匹配权重 key 加载（AutoWeightsLoader）。"""
        from vllm.model_executor.models.utils import AutoWeightsLoader
        return AutoWeightsLoader(self).load_weights(weights)
