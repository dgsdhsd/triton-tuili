from typing import Optional
import torch
import math
import torch.nn as nn
import os
import sys
import json
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
#======================================================================
#核函数列表
#======================================================================
@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, 
    k_ptr, v_ptr,
    start_n, end_n, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    offs_m, offs_n,
    STAGE: tl.constexpr, qk_scale
):
    """
    TMA 版内层循环，适配基 2 对数优化 (exp2)。
    - 使用原生 while 循环。
    - 不返回任何 k_ptr / v_ptr，切断外层 while 循环的 SSA 跨迭代依赖。
    - 入口处根据 start_n 自动完成 TMA 描述符的绝对位置推进对齐。
    """
    # 1. 强制强转 start_n，防止循环变量类型退化，并作为 while 循环的迭代器
    curr_n = tl.cast(start_n, tl.int32)

    # 2. 入口对齐推进：若 start_n > 0 (如 Stage 2)，一次性将原始指针推进到 start_n 位置
    # 注意：Triton 中对 TMA 指针的 advance 必须在真正的硬件指令前完成。
    if start_n > 0:
        k_ptr = tl.advance(k_ptr, [start_n, 0])
        v_ptr = tl.advance(v_ptr, [start_n, 0])

    # # 3. 原生 while 循环：条件非常干净，依赖 curr_n 和 end_n
    while curr_n < end_n:
    #     # 1) 直接使用 TMA 指针读取 K, V
        k = tl.load(k_ptr, boundary_check=(0, 1))
        v = tl.load(v_ptr, boundary_check=(0, 1))

        # 2) 计算 QK^T 并应用 scale
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk *= qk_scale  # 此时 qk 已经在 log2 尺度下

        # 3) Ragged 尾部数学掩码 (显式广播 + FP32 安全极小值)
        mask_1d = (curr_n + offs_n)[None, :] < seq_len           # Shape: [1, BLOCK_N]
        seq_mask = tl.broadcast_to(mask_1d, [BLOCK_M, BLOCK_N])  # Shape: [BLOCK_M, BLOCK_N]
        qk = tl.where(seq_mask, qk, -1.0e20)

        # 4) 因果掩码 (STAGE == 2)
        if STAGE == 2:
            # 绝对逻辑坐标比对
            causal_mask = offs_m[:, None] >= (curr_n + offs_n)[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)

        # 5) Base-2 Online Softmax 与 V 累加
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp2(m_i - m_ij)

        l_i = l_i * alpha + l_ij  #崩溃起始点
        # acc = acc * alpha[:, None]

        # # 执行 V 的矩阵乘法
        # acc = tl.dot(p.to(v.dtype), v, acc)
        # m_i = m_ij

    #     # 6) 单步长指针与逻辑坐标推进 (规整的线性递增，Triton 可完美推导)
        curr_n += BLOCK_N
        k_ptr = tl.advance(k_ptr, [BLOCK_N, 0])
        v_ptr = tl.advance(v_ptr, [BLOCK_N, 0])

    # 7. 坚决不返回 k_ptr, v_ptr
    return acc, l_i, m_i

@triton.jit
def _ragged_attn_fwd_kernel_persistent(
    Q, K, V, Out, M,
    cu_tiles, cu_seqlens, tile2seq_ptr, tile_counter_ptr,
    sm_scale,
    stride_qz, stride_qm, stride_qh,#0，512，64
    stride_kz, stride_km, stride_kh,
    stride_vz, stride_vm, stride_vh,
    stride_oz, stride_om, stride_oh,
    total_tasks, NUM_HEADS, NUM_KV_HEADS, MAX_TILE_M,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 抢占任务 (Grouped GEMM 范式)
    task_id = tl.atomic_add(tile_counter_ptr, 1)

    while task_id < total_tasks:
        # --- 任务解算与物理映射 ---
        head_idx = task_id % NUM_HEADS
        tile_m_and_seq = task_id // NUM_HEADS
        

        seq_idx = tl.load(tile2seq_ptr + tile_m_and_seq)
                
        seq_tile_start = tl.load(cu_tiles + seq_idx)
        tile_m_idx = tile_m_and_seq - seq_tile_start
        seq_tok_start = tl.load(cu_seqlens + seq_idx)
        seq_len = tl.load(cu_seqlens + seq_idx + 1) - seq_tok_start
        kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)

        # 依然保留 offs_m / n 用于后续的因果掩码 (Causal Mask) 数学计算
        offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        # =======================================================
        # TMA Block Pointers 构造区
        # =======================================================
        # 绝妙点：我们将 shape 的第一维限制为当前 sequence 的全局物理终止位置。
        seq_end_offset = seq_tok_start + seq_len

        # --- Q 的 TMA 指针与加载 ---
        q_ptr = tl.make_block_ptr(
            base=Q + head_idx * stride_qh,       # 指向当前 Head 的全局基址
            shape=(seq_end_offset, BLOCK_K),     # 硬件边界只到当前 Sequence 结尾
            strides=(stride_qm, 1),              # 物理内存步长
            offsets=(seq_tok_start + tile_m_idx * BLOCK_M, 0), # 定位到当前 Q 块
            block_shape=(BLOCK_M, BLOCK_K),      # 期望读取的块大小 (如 64x64)
            order=(1, 0)                         # 行优先布局
        )

        # TMA 硬件加载，boundary_check=(0, 1) 告诉硬件对超出 seq_end_offset 的部分补 0
        q = tl.load(q_ptr, boundary_check=(0, 1))

        # --- K 和 V 的 TMA 指针 (传给 inner loop) ---
        # 注意：内层循环起始位置为 seq_tok_start，会在 _attn_fwd_inner 中使用 tl.advance 逐步向后推
        k_ptr = tl.make_block_ptr(
            base=K + kv_head_idx * stride_kh,
            shape=(seq_end_offset, BLOCK_K),
            strides=(stride_km, 1),
            offsets=(seq_tok_start, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0)
        )
        
        v_ptr = tl.make_block_ptr(
            base=V + kv_head_idx * stride_vh,
            shape=(seq_end_offset, BLOCK_K),
            strides=(stride_vm, 1),
            offsets=(seq_tok_start, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0)
        )

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)#预留...........
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        qk_scale = sm_scale*1.44269504

        # =======================================================
        # Stage 控制流调度 (注意我们把 k_base/v_base 换成了 TMA 的 k_ptr/v_ptr)
        # =======================================================
        if IS_CAUSAL:
            # Stage 1: Off-band
            safe_end_n = tl.minimum(tile_m_idx * BLOCK_M, seq_len)
            # 【修改点】：直接丢弃 k_ptr, v_ptr 返回值，保持 outer 层的指针不发生写回
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, k_ptr, v_ptr, 
                0, safe_end_n, seq_len, BLOCK_M, BLOCK_K, BLOCK_N,
                offs_m, offs_n, 
                STAGE=1, qk_scale=qk_scale
            )
    
            # Stage 2: On-band 
            danger_end_n = tl.minimum((tile_m_idx + 1) * BLOCK_M, seq_len)
            if safe_end_n < seq_len:
                # 【修改点】：传入原始的 k_ptr, v_ptr，同样不接收指针返回值
                acc, l_i, m_i = _attn_fwd_inner(
                    acc, l_i, m_i, q, k_ptr, v_ptr, 
                    safe_end_n, danger_end_n, seq_len, BLOCK_M, BLOCK_K, BLOCK_N,
                    offs_m, offs_n, 
                    STAGE=2, qk_scale=qk_scale
                )
        else:
            # BERT Dense 模式
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, k_ptr, v_ptr, 
                0, seq_len, seq_len, BLOCK_M, BLOCK_K, BLOCK_N,
                offs_m, offs_n, 
                STAGE=1, qk_scale=qk_scale
            )

        # --- TMA 安全写回 ---
        acc = acc / l_i[:, None]
        
        o_ptr = tl.make_block_ptr(
            base=Out + head_idx * stride_oh,
            shape=(seq_end_offset, BLOCK_K),
            strides=(stride_om, 1),
            offsets=(seq_tok_start + tile_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        
        # TMA 硬件级 Mask 写入，绝不会越界污染下一条序列
        tl.store(o_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))

        # 继续抢下一个任务
        task_id = tl.atomic_add(tile_counter_ptr, 1)


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
class FlashAttentionWrapper(torch.autograd.Function):
    """
    负责将高层 Q, K, V 张量及硬件 Launch 参数适配并送入底层的 Triton _attention 核函数。
    """
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,             # [Total_Tokens, NUM_HEADS, HEAD_DIM]
        k: torch.Tensor,             # [Total_Tokens, NUM_KV_HEADS, HEAD_DIM]
        v: torch.Tensor,             # [Total_Tokens, NUM_KV_HEADS, HEAD_DIM]
        cu_seqlens: torch.Tensor,    # [NUM_SEQS + 1] int32 累加 Token 长度
        max_seqlen: int,             # 当前 Batch 中的最大 Sequence 长度
        causal: bool = True,
        sm_scale: float = None,
        warp_specialize: bool = True # Triton 3.x / Hopper 特性
    ):
        # ---------------------------------------------------------
        # 1. 基础张量校验与参数确定
        # ---------------------------------------------------------
        assert q.is_cuda and k.is_cuda and v.is_cuda and cu_seqlens.is_cuda
        assert q.dtype == k.dtype == v.dtype, "Q/K/V 必须保持相同的数据类型"
        assert cu_seqlens.dtype == torch.int32, "cu_seqlens 必须是 int32 类型"
        
        # 张量形状提取
        total_q_tokens, num_heads, head_dim = q.shape
        _, num_kv_heads, _ = k.shape
        
        # 默认 Scaling Factor
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)

        # 针对 TMA 的 Block Size 推荐配置 (Hopper 架构下 64x64 或 128x64 吞吐最佳)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = head_dim  # TMA 要求 Head Dim 一次性完全加载，BLOCK_K 必须等于 HEAD_DIM

        # TMA 要求内存地址必须满足 16-byte 对齐 (Float16 下 Head Dim 需为 8 的倍数)
        assert head_dim in {32, 64, 128, 256}, f"TMA 模式下 Head Dim ({head_dim}) 必须是 32/64/128/256"

        # ---------------------------------------------------------
        # 2. GPU 端并行构建 Persistent Task 映射表 (cu_tiles)
        # ---------------------------------------------------------
        # 计算每个 Sequence 被切分成了多少个 M-Tile
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        tiles_per_seq = (seqlens + BLOCK_M - 1) // BLOCK_M
        
        # 全 GPU 向量化计算 tile 的前缀和 (cu_tiles)
        cu_tiles = torch.zeros_like(cu_seqlens)
        cu_tiles[1:] = torch.cumsum(tiles_per_seq, dim=0)
        
        # 获取总 Task 数 (Tile 数量 * Head 数量)
        total_tiles = cu_tiles[-1].item()
        total_tasks = total_tiles * num_heads

        num_seqs = cu_seqlens.numel() - 1
        seq_ids = torch.arange(num_seqs, device=q.device, dtype=torch.int32)
        tile2seq_ptr = torch.repeat_interleave(seq_ids, tiles_per_seq)

        # ---------------------------------------------------------
        # 3. 输出缓冲区分配 & 原子计数器准备
        # ---------------------------------------------------------
        out = torch.empty_like(q)
        
        # Persistent 调度专用的全局原子计数器 (初始化为 0)
        tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)

        # ---------------------------------------------------------
        # 4. 网格 (Grid) 尺寸计算与 Kernel 启动
        # ---------------------------------------------------------
        # 获取当前 GPU 的 SM 物理核心数
        device_id = q.device.index if q.device.index is not None else 0
        num_sms = torch.cuda.get_device_properties(device_id).multi_processor_count
        
        # Persistent Grid 策略：常驻 SM 数量 2~4 倍的 CTA，让 SM 满载抢占任务
        grid_size = min(num_sms * 2, total_tasks)
        grid = (grid_size, )

        # 开启 Triton 3.x 编译优化参数
        extra_kwargs = {}
        if warp_specialize:
            # 在 Hopper (SM90+) 架构上开启 Warp Specialization 提升 TMA 搬运效率
            extra_kwargs["num_warps"] = 8
            extra_kwargs["num_stages"] = 3
        else:
            extra_kwargs["num_warps"] = 4
            extra_kwargs["num_stages"] = 2

        # 启动 TMA Persistent Kernel
        _ragged_attn_fwd_kernel_persistent[grid](
            Q=q, K=k, V=v, Out=out, M=None, # 若需要 Backward 可传 LogSumExp 缓冲区
            cu_tiles=cu_tiles,
            cu_seqlens=cu_seqlens,
            tile2seq_ptr=tile2seq_ptr,
            tile_counter_ptr=tile_counter,
            sm_scale=sm_scale,
            # 传入物理 Stride 用于 TMA 描述符配置
            stride_qz=0, stride_qm=q.stride(0), stride_qh=q.stride(1),
            stride_kz=0, stride_km=k.stride(0), stride_kh=k.stride(1),
            stride_vz=0, stride_vm=v.stride(0), stride_vh=v.stride(1),
            stride_oz=0, stride_om=out.stride(0), stride_oh=out.stride(1),
            total_tasks=total_tasks,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_kv_heads,
            MAX_TILE_M=(max_seqlen + BLOCK_M - 1) // BLOCK_M,
            IS_CAUSAL=causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            **extra_kwargs
        )

        # 保存反向传播所需的 Context Tensor
        ctx.save_for_backward(q, k, v, out, cu_seqlens, cu_tiles)
        ctx.sm_scale = sm_scale
        ctx.causal = causal

        return out

    @staticmethod
    def backward(ctx, do):
        pass

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
        # 1. Router 打分与 Top-K 选择
        # =====================================================================
        router_logits = self.router(x)  # [N, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1) # [N, top_k]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # 归一化 [N, top_k]

        # =====================================================================
        # 2. Permute: 根据选中专家对 Token 展平并按 Expert ID 排序
        # =====================================================================
        flat_topk_indices = topk_indices.view(-1)  # [N * top_k]
        flat_x = x.repeat_interleave(self.top_k, dim=0) # [N * top_k, d_model]

        # 排序：将发往相同专家的 Token 排在连续的内存区域
        sorted_expert_ids, permute_indices = torch.sort(flat_topk_indices)
        permuted_x = flat_x[permute_indices] # [N * top_k, d_model]

        # 计算每个专家分到的 Token 数量
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)

        # =====================================================================
        # 3. 调用 Triton Grouped GEMM 执行专家计算
        # =====================================================================
        # 3.1 按专家拆分输入输入 Tensor 列表 group_A_w1
        group_A_w1 = torch.split(permuted_x, expert_counts.tolist(), dim=0)
        # 取出所有的专家 W1 和 W2 矩阵列表
        group_B_w1 = [self.w1[e] for e in range(self.num_experts)] # self.w1: [num_experts, d_model, d_ffn]
        group_B_w2 = [self.w2[e] for e in range(self.num_experts)] # self.w2: [num_experts, d_ffn, d_model]

        # 3.2 过滤掉没有分到 Token (M=0) 的专家，防止 Triton 访问非法指针
        active_A_w1 = []
        active_B_w1 = []
        active_B_w2 = []
        active_counts = []

        for e in range(self.num_experts):
            if expert_counts[e] > 0:
                active_A_w1.append(group_A_w1[e])
                active_B_w1.append(group_B_w1[e])
                active_B_w2.append(group_B_w2[e])
                active_counts.append(expert_counts[e].item())

        if len(active_A_w1) > 0:
            # Step A: 第一次 GEMM -> Up-Projection [M_e, d_model] x [d_ffn, d_model]^T -> [M_e, d_ffn]
            group_C_w1 = group_gemm_tma_fn(active_A_w1, active_B_w1)

            # Step B: 激活函数 SiLU
            active_A_w2 = [F.silu(c_mat) for c_mat in group_C_w1]

            # Step C: 第二次 GEMM -> Down-Projection [M_e, d_ffn] x [d_model, d_ffn]^T -> [M_e, d_model]
            group_C_w2 = group_gemm_tma_fn(active_A_w2, active_B_w2)

            # 将活跃专家的计算结果重新拼接回 [N * top_k, d_model]
            permuted_out = torch.cat(group_C_w2, dim=0)
        else:
            permuted_out = torch.zeros_like(permuted_x)

        # 4. Unpermute: 还原 Token 原始顺序并加权求和
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
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 2,  # GQA 支持 (如 8 Q heads 共享 2 KV heads)
        head_dim: int = 64,
        num_experts: int = 8,
        top_k: int = 2,
        d_ffn: int = 2048,
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

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """
        x: [N, d_model] - 多个变长序列打平拼接后的输入
        cu_seqlens: [batch_size + 1] - 变长序列累积长度索引，如 [0, 60, 160, 410]
        max_seqlen: int - 当前 Batch 中最长句子的 Token 数
        """
        N, d_model = x.shape

        # ==========================================
        # 1. Attention 子层 (+ 残差连接)
        # ==========================================
        residual = x
        x_norm = self.input_norm(x)

        # (1) QKV 线性投影
        q = self.q_proj(x_norm).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_norm).view(N, self.num_kv_heads, self.head_dim)
        #self.q_proj(x_norm)完成X*WQ，.view(N, self.num_heads, self.head_dim)拆分三维

        # (2) 调用自写 FlashAttention 驱动
        attn_out = FlashAttentionWrapper.apply(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            True,   # 对应 causal
            None,   # 对应 sm_scale
            True    # 对应 warp_specialize
        )

        # (3) 输出拼接与投影
        attn_out = attn_out.view(N, self.num_heads * self.head_dim)
        attn_out = self.out_proj(attn_out)
        #多头信息融合：FlashAttention 计算完后，各个 Head 的注意力特征是独立分开的。乘以 $W_O$ 矩阵可以把所有 Head 抽取到的特征进行混合与信息融合。

        # (4) 残差相加
        x = residual + attn_out

        # ==========================================
        # 2. MoE 子层 (+ 残差连接)
        # ==========================================
        residual = x
        x_norm = self.post_attn_norm(x)

        # (1) MoE 专家计算 (Router -> Permute -> Grouped GEMM -> Unpermute)
        moe_out = self.moe_layer(x_norm)

        # (2) 残差相加
        x = residual + moe_out

        return x


# =====================================================================
# 2. 外部拉取逻辑：严格加载工具
# =====================================================================
def load_model_config(config_path: str) -> dict:
    """加载架构 JSON"""
    if not os.path.exists(config_path):
        print(f"❌ 严重错误：未找到配置 JSON '{config_path}'，拒绝启动！")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_weights(model: nn.Module, weights_path: str):
    """加载具体权重 (W) 的二进制文件"""
    if not os.path.exists(weights_path):
        print(f"❌ 严重错误：未找到权重文件 '{weights_path}'，拒绝启动！")
        print("因为所有参数现在是 empty，必须加载真实权重。")
        sys.exit(1)
    
    print(f"正在从 {weights_path} 加载权重...")
    # 使用 mmap 映射或直接加载到设备
    state_dict = torch.load(weights_path, map_location=next(model.parameters()).device, weights_only=True)
    
    # 严格模式：权重的 key 必须和代码定义的变量完全匹配
    model.load_state_dict(state_dict, strict=True)
    print("✅ 权重加载成功！")

# =====================================================================
# 3. 主函数执行与测试
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 文件路径统一定义
    config_file_path = "/home/xkem/cuda_triton/project/parameter/model_config.json"
    weights_file_path = "/home/xkem/cuda_triton/project/parameter/model_weights.pt"

    # [拉取 1]：读取 JSON 构建架构
    model_kwargs = load_model_config(config_file_path)
    model = GroupTransformer(**model_kwargs).to(device)

    # [拉取 2]：将 PT 权重文件灌入刚刚构建好的空模型
    # 注意：如果此时你还没有权重文件，这里会因为严格模式直接报错停止。
    load_model_weights(model, weights_file_path)

    # [动态生成]：构造运行时数据
    seqlens = [60, 100, 250,480,950,2000]
    cu_seqlens = torch.tensor([0, 60, 160, 410, 890, 1840, 3840], dtype=torch.int32, device=device)
    max_seqlen = max(seqlens)
    total_tokens = cu_seqlens[-1].item()
    x = torch.randn(total_tokens, model_kwargs["d_model"], dtype=torch.float16, device=device)

    # 执行前向传播
    model.eval()
    with torch.no_grad():
        out = model(x, cu_seqlens, max_seqlen)

    print("输入 Tensor 形状:", x.shape)
    print("输出 Tensor 形状:", out.shape)