#!/usr/bin/env bash
# =============================================================================
# DeepSeek-Coder-V2-Lite 多卡分布式推理启动脚本（TP + EP 核心示例）
#
# 目标：用原生 `vllm serve` 拉起 OpenAI 兼容 API，体现完整的分布式并行拓扑。
# 本机（单卡）无法真正并行，但本脚本参数即"未来上云多卡"直接可用的配置。
#
# 并行维度一览（乘积 = 使用 GPU 总数）：
#   - TP 张量并行：每层线性层按列/行切分，MLA 注意力按 head 切，通信=all-reduce
#   - EP 专家并行：MoE 的 64 个 routed expert 分布到 TP 卡，通信=all-to-all
#   - PP 流水线并行：27 层按段切分，通信=点对点 send/recv
#   - DP 数据并行：多副本，吞吐扩展（vLLM 0.29 起支持 -dp）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# ---------------------------------------------------------------------------
# 1) 分布式并行拓扑（按云上 GPU 数量修改）
# ---------------------------------------------------------------------------
TP=4        # 张量并行度。MoE 下同时是 EP 的"总卡数"（EP 在 TP 组内切专家）
PP=1        # 流水线并行度。本模型只有 27 层、hidden 2048，浅模型 PP=1 即可
DP=1        # 数据并行副本数。多副本用 -dp，配合负载均衡
ENABLE_EP=1 # 是否开启专家并行（DeepSeek-V2 MoE 的核心并行维度）

# 三者乘积 = GPU 总数。例如 8 卡可选：TP=8/EP=8, PP=1, DP=1
GPU_TOTAL=$((TP * PP * DP))
echo "[topo] TP=${TP} PP=${PP} DP=${DP} EP=$([ "$ENABLE_EP" = 1 ] && echo on || echo off) -> 共 ${GPU_TOTAL} 卡"

# ---------------------------------------------------------------------------
# 2) 模型来源（三选一，取消注释对应行）
# ---------------------------------------------------------------------------
# A. 官方 BF16（多卡标准选择，32GB 权重由 TP+EP 分片到各卡，约 8GB/卡）：
MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
# B. AWQ 4bit（小显存/单卡，8.5GB；见 serve_single.sh）：
# MODEL="TechxGenus/DeepSeek-Coder-V2-Lite-Base-AWQ"
# C. 本地目录（只含 config.json、无权重，仅用于架构自检，需配 --load-format dummy）：
# MODEL="$(pwd)/model"

# ---------------------------------------------------------------------------
# 3) 组装启动参数
# ---------------------------------------------------------------------------
EP_FLAG=""
if [ "$ENABLE_EP" = "1" ]; then
  EP_FLAG="--enable-expert-parallel"   # 把 64 个 expert 分布到 TP 卡，token 用 all-to-all 路由
fi

# C 模式（本地无权重）时，取消下一行注释，用随机权重走通模型构建：
# DUMMY_FLAG="--load-format dummy"
DUMMY_FLAG=""

# 云上更进阶的可选项（需要时取消注释）：
#   --enable-eplb                # 专家并行负载均衡（把热门专家冗余到多卡）
#   --eplb-config '{"num_redundant_experts": 8}'
#   --all2all-backend deepep     # EP 的 all-to-all 通信后端（deep_ep/standard）
#   --expert-placement-strategy round_robin

exec /home/xkem/cuda_triton/vllm_env/bin/vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --data-parallel-size "$DP" \
  $EP_FLAG \
  $DUMMY_FLAG \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --served-model-name deepseek-coder-v2-lite \
  --host 0.0.0.0 \
  --port 8000
