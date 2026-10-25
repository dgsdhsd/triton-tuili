# TP / EP 源码阅读地图（DeepSeek-Coder-V2-Lite）

> 本文件是"看源码学分布式"的导航。源码分两层：
> 1. **vLLM 推理实现**（含 TP/EP/PP 分布式代码）—— 学习分布式的核心，已装在本机；
> 2. **transformers 架构实现**（纯 PyTorch 单卡版）—— 理解模型结构本身（MLA/MoE）。
>
> 建议先看第 2 层理解"模型长什么样"，再看第 1 层理解"如何切开并行"。

---

## 第一层：vLLM 推理实现（含分布式）

主文件：
`/home/xkem/cuda_triton/vllm_env/lib/python3.10/site-packages/vllm/model_executor/models/deepseek_v2.py`
（约 1550 行）

### 1. TP 张量并行

**并行线性层（import，第 67–72、91 行）**：
- `ColumnParallelLinear`：按**输出维**切分（每卡算一部分列），输入各卡复制，输出需 all-gather。
- `RowParallelLinear`：按**输入维**切分，输出需 all-reduce 求和。
- `MergedColumnParallelLinear`：多个列并行线性层合并（如 gate+up 一起切）。
- `VocabParallelEmbedding`：embedding 按 vocab 切分。

**通信算子**：
- `tensor_model_parallel_all_gather`（第 424、1356、1529、1553 行）
- `tensor_model_parallel_reduce_scatter`（第 1383 行）

**在哪些层用到**：
- MLA 注意力 `DeepseekV2Attention`（约第 478–529 行）：`q_b_proj`/`kv_b_proj` 用 `ColumnParallelLinear`（按 head 切），`o_proj` 用 `RowParallelLinear`。
- 共享专家 MLP（第 257、265 行）：`gate_up_proj` = `MergedColumnParallelLinear`，`down_proj` = `RowParallelLinear`。
- `DeepSeekV2FusedQkvAProjLinear`（第 943 行）：把 q/kv 投影融合后列并行。

> 阅读要点：看每个 `ColumnParallelLinear` 的 `output_size` 如何除以 `tp_size`，以及 forward 后紧跟的 all-gather/all-reduce 位置。

### 2. EP 专家并行（本模型的核心）

**主类 `DeepseekV2MoE`（第 287 行起）**，关键代码：
```python
self.ep_group = get_ep_group().device_group          # 第 303 行
self.ep_size  = self.ep_group.size()                 # 第 304 行
self.n_routed_experts = config.n_routed_experts      # 第 305 行（64）
...
self.n_physical_experts = n_logical_experts + n_redundant_experts  # 第 336 行
self.n_local_physical_experts = self.n_physical_experts // self.ep_size  # 第 337 行 ★EP 核心
```
- **第 337 行是 EP 的灵魂**：每张卡只持有 `64 / ep_size` 个专家。
- 第 370 行 `FusedMoEFactory(...)`：创建融合专家层，内部做 dispatch → 专家前向 → combine。
- 第 331–337 行：**EPLB**（专家并行负载均衡）——`enable_eplb`、`num_redundant_experts` 把热门专家冗余到多卡。
- 第 414–427 行：`sequence_parallel_chunk` / `all_gather`——序列并行优化 MoE。

**EP 的 all-to-all 通信不在本文件**，在独立目录：
`/home/xkem/cuda_triton/vllm_env/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/`

| 文件 | 作用 |
|---|---|
| `layer.py` | FusedMoE 主层：token 路由 → 专家计算 → 归约的主流程 |
| `all2all_utils.py` | **all-to-all 通信**（dispatch/combine 的具体实现） |
| `fused_moe.py` | 融合 MoE kernel（含 gptq/awq 量化 kernel，第 65 行 `fused_moe_kernel_gptq_awq`） |
| `moe_align_block_size.py` | 把 token 按 block 对齐（对齐后 all-to-all 更高效） |
| `topk_weight_and_reduce.py` | top-k 专家输出按权重归约 |
| `router/` | 路由 gate（`fused_topk_router.py` 等） |
| `config.py` | FusedMoE 配置（`max_num_tokens` 等） |

> 阅读要点：EP 的完整链路是
> `hidden_states → gate(路由) → moe_align_block_size → all2all(dispatch) → 本地专家前向 → all2all(combine) → topk加权归约`。
> 从 `layer.py` 的 forward 一路跟下去即可看清。

### 3. PP 流水线并行
- `get_pp_group().is_first_rank`（第 1433 行）：只有第一个 PP rank 放 embedding。
- `get_pp_group().is_last_rank`（第 1458 行）：只有最后一个 PP rank 放 norm + lm_head。
- `PPMissingLayer()`：非本 PP 段的层用占位符。
- `make_layers(...)`（第 1442 行）：按 PP 切分 `start_layer`/`end_layer`。

---

## 第二层：transformers 架构实现（纯 PyTorch，单卡）

来源：官方仓库 `deepseek-ai/DeepSeek-Coder-V2-Lite-Base` 的 `modeling_deepseek.py`
（`configuration_deepseek.py` 是配置类）。

要理解的核心类（标准 DeepSeek-V2 结构）：

| 类 | 作用 |
|---|---|
| `DeepseekV2RMSNorm` | RMSNorm（不用 LayerNorm） |
| `DeepseekV2Attention` | **MLA** 注意力：kv 压缩到 latent（kv_lora_rank=512），q 拆 nope/rope 两部分 |
| `DeepseekV2MLP` | 共享专家的 dense MLP |
| `DeepseekV2MoE` | MoE：gate 路由出 top-k 专家 → 各专家前向 → 加权求和 |
| `DeepseekV2DecoderLayer` | 单层：attention + MoE + 残差 |
| `DeepseekV2Model` / `DeepseekV2ForCausalLM` | 整体模型 / 带 lm_head |

> 对比阅读：transformers 版的 `DeepseekV2MoE.forward` 是"循环遍历激活的 6 个专家分别算"，
> 而 vLLM 版用 `FusedMoE` + all-to-all 一次性算完并跨卡并行——两者对照最能看懂 EP 的价值。

---

## 建议阅读顺序

1. `configuration`/`config.json`：先把 `n_routed_experts=64`、`num_experts_per_tok=6`、MLA 各维度记住。
2. transformers `modeling_deepseek.py` 的 `DeepseekV2MoE`：理解"一个 token 激活 6 个专家"意味着什么。
3. vLLM `deepseek_v2.py` 第 287–427 行 `DeepseekV2MoE`：看 `ep_size` 如何决定每卡专家数。
4. vLLM `fused_moe/layer.py` + `all2all_utils.py`：看 all-to-all 如何把 token 送到持有对应专家的卡。
5. 回到 `deepseek_v2.py` 的 `ColumnParallelLinear`/`RowParallelLinear` 用法：看 TP 如何切 attention/MLP。

---

## 附：如何用 gdb/日志验证并行确实发生（多卡云上）

```bash
# 启动时看拓扑日志（vLLM 会打印 world size、TP/EP 分组）
bash serve.sh 2>&1 | grep -iE 'parallel|world|expert|nccl'
# 用 NCCL 调试环境变量看 all-to-all / all-reduce 通信：
#   export NCCL_DEBUG=INFO
```
