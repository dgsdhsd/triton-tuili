#!/bin/bash
# 运行 vLLM 离线推理示例
# 用法: bash vllm_demo/run.sh [--model ...] [--max-tokens ...]
cd "$(dirname "$0")/.."

# 国内网络下 HuggingFace 用镜像站加速模型下载（已下载缓存后无影响）
export HF_ENDPOINT=https://hf-mirror.com

# 禁用 Xet 存储后端（hf-mirror 镜像不支持，否则报 401 Unauthorized）
export HF_HUB_DISABLE_XET=1

# WSL2 下 vLLM 默认禁用 pinned memory，需显式开启（内核 >= 4.19.121 时）
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# 禁用 FlashInfer 采样器（FlashInfer 0.6.x 对 sm_120/Blackwell 架构检测有 bug，
# 会误报 "requires sm75 or higher"，禁用后回退到 PyTorch 原生采样）
export VLLM_USE_FLASHINFER_SAMPLER=0

# 使用独立虚拟环境里的 python 运行，避免污染 cuda_triton 的 torch/triton 环境
exec ./vllm_env/bin/python vllm_demo/vllm_demo.py "$@"
