import pytest
import torch
import torch.nn as nn
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9

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
    """
    # TMA 指针 fast-forward 到当前起点
    k_ptr = tl.advance(k_ptr, [start_n, 0])
    v_ptr = tl.advance(v_ptr, [start_n, 0])


    while start_n < end_n:
        # 1. TMA 读取 K, V (零开销阻断越界物理访存)
        k = tl.load(k_ptr, boundary_check=(0, 1))
        v = tl.load(v_ptr, boundary_check=(0, 1))

        # 2. 计算 QK^T 并应用带有 1/ln(2) 的 scale
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale  # 此时 qk 已经在 log2 尺度下了

        # 3. Ragged 尾部数学掩码 (必加，防止越界处 exp2(0)=1 污染分母)
        seq_mask = (start_n + offs_n[None, :]) < seq_len
        qk = tl.where(seq_mask, qk, float("-inf"))

        # 4. 因果掩码 (STAGE == 2)
        if STAGE == 2:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # 5. Base-2 Online Softmax 与 V 累加
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        
        # 【核心修改点】：tl.exp -> tl.exp2
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # 【核心修改点】：tl.exp -> tl.exp2
        alpha = tl.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # 执行 V 的矩阵乘法
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

        # 6. 滑动窗口与指针推进
        start_n += BLOCK_N
        k_ptr = tl.advance(k_ptr, [BLOCK_N, 0])
        v_ptr = tl.advance(v_ptr, [BLOCK_N, 0])

    return acc, l_i, m_i

@triton.jit
def _ragged_attn_fwd_kernel_persistent(
    Q, K, V, Out, M,
    cu_tiles, cu_seqlens, tile_counter_ptr,
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
        
        # 二分查找 seq_idx
        low, high = 0, cu_tiles.shape[0] - 1
        seq_idx = 0
        while low <= high:
            mid = (low + high) // 2
            if tl.load(cu_tiles + mid) <= tile_m_and_seq:
                seq_idx = mid; low = mid + 1
            else:
                high = mid - 1
                
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

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        qk_scale = sm_scale*1.44269504

        # =======================================================
        # Stage 控制流调度 (注意我们把 k_base/v_base 换成了 TMA 的 k_ptr/v_ptr)
        # =======================================================
        if IS_CAUSAL:
            # Stage 1: Off-band
            safe_end_n = tl.minimum(tile_m_idx * BLOCK_N, seq_len)
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, k_ptr, v_ptr, 
                0, safe_end_n, seq_len, BLOCK_M, BLOCK_K, BLOCK_N,
                offs_m, offs_n, 
                STAGE=1, qk_scale=qk_scale
            )
            
            # Stage 2: On-band 
            danger_end_n = tl.minimum((tile_m_idx + 1) * BLOCK_N, seq_len)
            if safe_end_n < seq_len:
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


class _attention(torch.autograd.Function):

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
        a=10


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, flash_attn_fn):
        """
        d_model: 隐藏层特征维度 (如 512 或 4096)
        num_heads: 注意力头数 (如 8 或 32)
        flash_attn_fn: 你之前手写并测试好的 Triton Persistent FlashAttention 内核函数
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除！"
        self.head_dim = d_model // num_heads
        
        # 保存手写的 Triton 算子句柄
        self.flash_attn_fn = flash_attn_fn

        # 1. 输入投影层 (也可以打包成一个 nn.Linear(d_model, 3 * d_model))
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 2. 输出投影层 W_O
        self.out_proj = nn.Linear(d_model, d_model, bias=False)


def generate_ragged_test_inputs(
    seqlens: list[int] = [64, 128, 256, 512, 1024, 2048], # 齐整的 2 的 n 次方长度
    num_heads: int = 8,
    head_dim: int = 64, # K 保持 64
    dtype = torch.float16,
    device = DEVICE
):
    """
    生成符合变长 (Ragged) 算子接口的标准齐整输入数据。
    Q, K, V, dO 在 HBM 内存中打平连续存放，
    cu_seqlens 和 max_seqlen 在外部构建完毕后传入。
    """
    total_tokens = sum(seqlens)
    max_seqlen = max(seqlens)

    # 1. 在 HBM 中构造连续打平的 Q, K, V, dO
    # 形状: [Total_Tokens, Num_Heads, Head_Dim]
    q = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=dtype, requires_grad=True)
    
    # 反向传播梯度输入 dO，形状与输出 O 一致
    do = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=dtype)

    # 2. 构造前缀和数组 cu_seqlens
    # 形状: [len(seqlens) + 1]，类型必须为 int32 Tensor 位于 GPU
    cu_seqlens_py = [0]
    for l in seqlens:
        cu_seqlens_py.append(cu_seqlens_py[-1] + l)
    cu_seqlens = torch.tensor(cu_seqlens_py, device=device, dtype=torch.int32)

    return q, k, v, do, cu_seqlens, max_seqlen

attention = _attention.apply

