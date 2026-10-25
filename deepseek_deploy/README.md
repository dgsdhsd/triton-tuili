# DeepSeek-Coder-V2-Lite 分布式部署框架（TP + EP）

> 目标：为**学习分布式推理（张量并行 TP / 专家并行 EP）**准备一套完整的 vLLM 部署框架。
> 本机只有单张 RTX 5070（12GB），无法真正多卡并行；本仓库是"配置即文档"——
> 脚本参数可直接搬到多卡云服务器运行，源码阅读地图已整理好。

---

## 一、为什么选 DeepSeek-Coder-V2-Lite（16B）

它是标准的 **MoE（混合专家）+ MLA（多头潜在注意力）** 架构，是学习分布式的最佳对照物：

| 结构 | 数值 | 对分布式学习的意义 |
|---|---|---|
| `n_routed_experts` | 64 | **EP 专家并行**的切分对象（64 个专家分布到多卡） |
| `n_shared_experts` | 2 | 共享专家，每个 token 都走，不参与 EP |
| `num_experts_per_tok` | 6 | 每个 token 激活 6 个专家，需 **all-to-all** 路由 |
| `num_hidden_layers` | 27 | **PP 流水线并行**的切分对象 |
| `hidden_size` | 2048 | 每层线性层是 **TP 张量并行**的切分对象 |
| MLA（kv_lora_rank=512） | — | KV cache 极小，TP 按 head 切投影 |

**TP / EP 不是模型权重自带的，而是 vLLM 运行时施加的并行策略**：
- 权重文件只是一堆张量；
- 你启动时给 `--tensor-parallel-size N --enable-expert-parallel`，vLLM 在 `deepseek_v2.py` 里动态切分权重并插入通信（TP=all-reduce，EP=all-to-all）。

---

## 二、目录结构

```
deepseek_deploy/
├── README.md               # 本文件
├── env.sh                  # 公共环境变量（网络镜像 / WSL2 / Blackwell 兼容）
├── serve.sh                # ★ 多卡 TP+EP 启动脚本（核心，上云直接改拓扑）
├── serve_single.sh         # 单卡调试版（AWQ 4bit）
├── test_api.py             # OpenAI SDK 接口测试（completions + chat）
├── test_curl.sh            # curl 底层接口测试
├── check_config.py         # 配置自检（无需权重，秒级验证配置合法）
├── model/
│   └── config.json         # 官方 BF16 架构配置（无权重文件）
└── docs/
    └── tp_ep_source_map.md # ★ TP/EP 源码阅读地图（含行号）
```

---

## 三、关键概念速览（并行四维）

| 维度 | 全称 | 切分对象 | 通信算子 | vLLM 参数 |
|---|---|---|---|---|
| **TP** | Tensor Parallel | 每层线性层按列/行切，MLA 按 head 切 | all-reduce / all-gather | `--tensor-parallel-size` |
| **EP** | Expert Parallel | MoE 的 64 个专家分布到多卡 | all-to-all（dispatch/combine） | `--enable-expert-parallel` |
| **PP** | Pipeline Parallel | 27 层切成多段 | 点对点 send/recv | `--pipeline-parallel-size` |
| **DP** | Data Parallel | 多副本 | all-reduce（梯度/负载均衡） | `--data-parallel-size` |

GPU 总数 = TP × PP × DP。DeepSeek-V2 里 **EP 在 TP 组内**切分专家，因此 `--enable-expert-parallel` 要求 TP > 1。

---

## 四、使用方法

### 0. 环境
- 依赖独立虚拟环境 `/home/xkem/cuda_triton/vllm_env`（已装 vLLM 0.29.0、torch 2.11+cu128）。
- 本机 1 卡；云上多卡需保证 vLLM 版本一致 + NCCL 可用。

### 1. 配置自检（无需权重，先跑这个）
```bash
cd /home/xkem/cuda_triton/deepseek_deploy
/home/xkem/cuda_triton/vllm_env/bin/python check_config.py
```
验证：架构可被 vLLM 识别、MoE/MLA 字段齐全、64 专家可被 EP 整除。

### 2. 本机单卡调试（AWQ 4bit，可选下载权重）
```bash
bash serve_single.sh          # 首次自动下载约 8.5GB AWQ 权重
# 另开终端：
bash test_curl.sh             # 或 python test_api.py
```

### 3. 多卡云部署（TP + EP，核心目标）
```bash
# 编辑 serve.sh 顶部 TP/PP/DP 拓扑，例如 8 卡：
#   TP=8  ENABLE_EP=1  PP=1  DP=1
bash serve.sh                 # 首次自动下载官方 BF16（32GB）
```

---

## 五、模型来源说明（三选一）

| 场景 | 模型 ID | 精度/大小 | 说明 |
|---|---|---|---|
| 多卡标准 | `deepseek-ai/DeepSeek-Coder-V2-Lite-Base` | BF16 / 32GB | 官方权重，TP+EP 分片到各卡 |
| 单卡/小显存 | `TechxGenus/DeepSeek-Coder-V2-Lite-Base-AWQ` | AWQ 4bit / 8.5GB | 社区量化版 |
| 纯学配置 | `./model`（本目录） | 无权重 | 仅 config.json，配 `--load-format dummy` 随机权重自检 |

> **Base 是代码补全模型（无对话模板）**，接口用 `/v1/completions`；
> 若要真正对话，换 `...-Instruct` 版，接口用 `/v1/chat/completions`。

---

## 六、接口说明（vLLM 原生 OpenAI 兼容 API）

| 接口 | 用途 | 适用模型 |
|---|---|---|
| `GET /v1/models` | 列模型 | 全部 |
| `POST /v1/completions` | 文本/代码补全 | Base（无 chat template） |
| `POST /v1/chat/completions` | 多轮对话 | Instruct |

源码阅读从这里开始：`docs/tp_ep_source_map.md`。
