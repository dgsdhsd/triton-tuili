#!/usr/bin/env bash
# =============================================================================
# 公共环境变量（所有部署脚本 `source ./env.sh` 引入）
# 这些是本机（WSL2 + RTX 5070/Blackwell + 国内网络）已验证过的必要项，
# 云服务器上如无对应问题可保留（多设置无害）。
# =============================================================================

# 1) 国内网络：HuggingFace 走镜像站（模型已缓存后无影响）
export HF_ENDPOINT=https://hf-mirror.com

# 2) 禁用 Xet 存储后端：hf-mirror 镜像不支持 xet，否则下载报 401
export HF_HUB_DISABLE_XET=1

# 3) WSL2 下 vLLM 默认禁用 pinned memory，需显式开启（内核 >= 4.19.121）
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# 4) 禁用 FlashInfer 采样器：FlashInfer 对 sm_120/Blackwell 架构检测有 bug，
#    会误报 "requires sm75 or higher"，禁用后回退 PyTorch 原生采样
export VLLM_USE_FLASHINFER_SAMPLER=0

# 5) （多卡可选）单进程多卡时关闭 V1 多进程 fork，避免与 NCCL 冲突
# export VLLM_ENABLE_V1_MULTIPROCESSING=0
