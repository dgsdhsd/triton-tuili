#!/usr/bin/env python3
"""
vLLM 离线批量推理示例（可直接运行）

这是一个最小但完整的 vLLM 离线推理脚本，展示了 vLLM 最核心的用法：
    1. 用 LLM 类加载模型（自动做 PagedAttention 分页 KV cache、连续批处理等优化）
    2. 用 SamplingParams 指定采样参数
    3. 用 llm.generate() 一次对多条 prompt 做批量推理

运行方式（在装好 vLLM 的虚拟环境里）：
    python vllm_demo.py                        # 使用默认模型 Qwen/Qwen2.5-0.5B-Instruct
    python vllm_demo.py --model /path/to/model # 指定本地模型目录
    python vllm_demo.py --max-tokens 128       # 修改生成长度
"""

import argparse
import os
import time

# ---- 运行前必须的环境变量（用 setdefault，不覆盖用户已有设置）----
# 1) 国内网络用 HuggingFace 镜像站下载模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 2) 禁用 Xet 存储后端：hf-mirror 镜像不支持 xet（否则会报 401 Unauthorized）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 3) WSL2 下 vLLM 默认禁用 pinned memory（导致 UVA 不可用、引擎启动失败），
#    内核 >= 4.19.121 时设置此变量启用（`uname -r` 查看，本机 6.18.33.2 满足）
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
# 4) 禁用 FlashInfer 采样器：FlashInfer 0.6.x 对 RTX 5070 (sm_120/Blackwell)
#    架构检测有 bug，会误报 "requires sm75 or higher"。禁用后回退到 PyTorch 原生采样。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def build_llm(model: str, gpu_memory_utilization: float):
    """加载模型。vLLM 会自动选择最优的注意力/GEMM 后端（如 FlashAttention、量化 kernel 等）。"""
    from vllm import LLM

    return LLM(
        model=model,
        # 单卡推理。多卡时可设 tensor_parallel_size=卡数 做张量并行
        tensor_parallel_size=1,
        # 允许 vLLM 使用的 GPU 显存比例（PagedAttention 的 KV cache 也从这里分配）
        gpu_memory_utilization=gpu_memory_utilization,
        # 最大序列长度（prompt + 生成），超出会被截断
        max_model_len=2048,
        # dtype 用 "auto"：让 vLLM 根据模型权重自动选 fp16/bf16
        dtype="auto",
        # 首次运行自动从 HuggingFace 下载模型权重；已下载过则直接用缓存
        trust_remote_code=True,
    )


def main():
    parser = argparse.ArgumentParser(description="vLLM 离线批量推理示例")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace 模型名或本地模型路径",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="每条回复最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度，越高越随机")
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="GPU 显存利用率 (0~1)")
    args = parser.parse_args()

    # ---- 1. 加载模型 ----
    print(f"==> 正在加载模型: {args.model}")
    llm = build_llm(args.model, args.gpu_mem)

    # ---- 2. 准备多条 prompt（一次批量推理，vLLM 会做 continuous batching）----
    prompts = [
        "用一句话解释什么是大语言模型。",
        "请用 Python 写一个计算斐波那契数列的函数。",
        "北京到上海的高铁大约需要多长时间？",
        "翻译成英文：今天天气真好。",
    ]

    # ---- 3. 指定采样参数 ----
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,  # 采样温度
        top_p=0.95,                    # nucleus sampling
        max_tokens=args.max_tokens,    # 每条回复最大长度
    )

    # ---- 4. 批量推理并计时 ----
    print(f"==> 开始对 {len(prompts)} 条 prompt 批量推理 ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    # ---- 5. 打印结果 ----
    for prompt, output in zip(prompts, outputs):
        generated = output.outputs[0].text
        print("\n" + "=" * 70)
        print(f"[提问] {prompt}")
        print(f"[回答] {generated}")

    # 打印吞吐统计
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print("\n" + "=" * 70)
    print(f"共 {len(prompts)} 条请求, 生成 {total_tokens} 个 token, "
          f"耗时 {elapsed:.2f}s, 吞吐 {total_tokens / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
