#!/usr/bin/env bash
# 用【新版】Nsight Systems 重新采集 MOE-LLM.py 的 GPU 时间轴。
# 背景：旧 nsys 2024.6.2 不支持 RTX 5070 (Blackwell sm_120)，GPU kernel 时间轴为空。
#
# 用法:
#   bash project/run_nsys_new.sh [nsys 可执行文件路径]
#   例:
#     bash project/run_nsys_new.sh /opt/nvidia/nsight-systems/2025.x/target-linux-x64/nsys
#     bash project/run_nsys_new.sh /home/xkem/nsight-systems/target-linux-x64/nsys
#   不传参数时，默认用 PATH 里的 nsys。
set -euo pipefail

NSYS="${1:-nsys}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="project/moe_llm_nsys"

echo "==> 使用的 nsys: $NSYS"
"$NSYS" --version

echo ""
echo "==> 方式1: 默认 CUDA trace (--trace=cuda)"
"$NSYS" profile -o "$OUT" --force-overwrite true \
    --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
    .venv/bin/python project/MOE-LLM.py

echo ""
echo "==> 校验是否采到 GPU kernel:"
"$NSYS" stats --report cuda_gpu_kern_sum --format csv \
    --output "${OUT}_kern" "$OUT.nsys-rep" 2>&1 | tail -4

if [ -s "${OUT}_kern_cuda_gpu_kern_sum.csv" ]; then
    echo "✅ GPU kernel 已采集到，时间轴可用。"
else
    echo "⚠️  默认 trace 下 GPU kernel 仍为空（Blackwell 硬件 trace 在 WSL2/VM 下可能失败）。"
    echo "==> 方式2: 软件插桩 trace (--trace=cuda-sw)，适配 WSL2/VM 环境"
    "$NSYS" profile -o "$OUT" --force-overwrite true \
        --trace=cuda-sw,nvtx,osrt \
        .venv/bin/python project/MOE-LLM.py
    "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
        --output "${OUT}_kern" "$OUT.nsys-rep" 2>&1 | tail -4
fi

echo ""
echo "✅ 完成。报告文件: $ROOT/$OUT.nsys-rep"
echo "   用 nsys-ui 打开即可查看完整时间轴计算图（或在 nsys 里看 kernel 汇总）。"
