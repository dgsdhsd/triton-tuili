#!/usr/bin/env bash
# =============================================================================
# 单卡调试版：TP=1，用 AWQ 4bit 量化（权重约 8.5GB，12GB 显存可装下）
#
# 与 serve.sh 的区别：本机只有 1 张 RTX 5070（12GB），装不下官方 BF16（32GB），
# 故退化为 TP=1 + AWQ 4bit。并行维度全部为 1，代码路径存在但不发通信。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# AWQ 4bit 量化版（首次运行会自动下载约 8.5GB）：
MODEL="TechxGenus/DeepSeek-Coder-V2-Lite-Base-AWQ"

# 若不想下载权重，改用本地 config 目录 + 随机权重做架构自检：
# MODEL="$(pwd)/model"
# LOAD_FLAG="--load-format dummy"
LOAD_FLAG=""

exec /home/xkem/cuda_triton/vllm_env/bin/vllm serve "$MODEL" \
  --quantization awq \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  $LOAD_FLAG \
  --trust-remote-code \
  --served-model-name deepseek-coder-v2-lite \
  --host 127.0.0.1 \
  --port 8000
