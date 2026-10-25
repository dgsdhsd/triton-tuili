#!/usr/bin/env python3
"""
vLLM 接口自测：方案 A 注册 SelfLLM，跑 prefill + decode。

重点：
  1. 验证 vLLM 调用 SelfLLM 的接口正确（__init__(vllm_config)、forward(input_ids, positions)、compute_logits）。
  2. 显现 prefill（长序列）与 decode（单 token）两种场景（见 SelfLLM.forward 的打印）。
  3. 自写 Triton attention（继承 vLLM Attention 层做 KV cache / metadata）+ MoE 已接入 GroupTransformer。
  4. KV cache 由 vLLM 自动分配（启动日志可见），暂不关心存储爆炸/调度。
"""
import importlib.util
import os
import sys

# ---- 环境变量 guard rails（WSL2 + RTX 5070 / Blackwell）----
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 让自定义注册/后端生效
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")     # 5070 上 FlashInfer 有 bug
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")     # WSL2 下启用 pinned memory（否则 UVA 不可用）
os.environ.setdefault("SELF_LLM_DEBUG", "1")                  # 打印 attention 的推理阶段（prefill/decode/chunked）


def load_self_llm_module():
    """用 importlib 加载 SELF-LLM.py（文件名带连字符，无法直接 import）。"""
    path = "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
    spec = importlib.util.spec_from_file_location("self_llm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["self_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_sampling_params(SamplingParams, max_tokens=8, temperature=0.8):
    """统一采样参数：detokenize=False（无 tokenizer）、ignore_eos=True（无 EOS）。"""
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        ignore_eos=True,     # 没有 tokenizer 就没有 EOS，强制生成满 max_tokens
        detokenize=False,    # 直接输出 token ids
    )


def main():
    # ---- 1. 加载 SelfLLM 类 + 方案 A 注册 ----
    self_llm = load_self_llm_module()
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("SelfLLM", self_llm.SelfLLM)

    # ---- 2. 加载引擎（skip_tokenizer_init：直接喂 token ids；dummy：随机权重）----
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/home/xkem/cuda_triton/vllm_demo/models/self-llm",
        skip_tokenizer_init=True,    # 不用 tokenizer，直接喂 token ids
        load_format="dummy",         # 权重随机，不加载真实文件
        trust_remote_code=True,      # 允许从模型目录加载 configuration_self_llm.py
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,  # 给 KV cache 留显存（启动日志可见分配）
        max_model_len=128,
        enforce_eager=True,
        max_num_batched_tokens=32,   # 调小（默认 8192）：让 >32 token 的 prompt 触发 chunked prefill
    )

    # ---- 3. 定义「推理阶段」测试 ----
    # 关键：max_num_batched_tokens 调小到 32（默认 8192），让 >32 token 的 prompt 触发 chunked prefill。
    # 各阶段含义（对应 SELF-LLM.py 里 attention 核的不同执行路径）：
    #   - 纯 prefill       ：全新请求，无历史（prefix_len=0），走 prefill 核 Stage B
    #   - 多条 prefill     ：多个全新请求同批 prefill
    #   - chunked prefill  ：长 prompt 被切成多个 chunk，第一个 chunk 是纯 prefill，
    #                        后续 chunk 就是 prefix-prefill（Stage A 读历史 + Stage B 算新词）
    #   - decode           ：生成阶段，每次 1 token（每条序列），走 decode 核
    stages = [
        ("纯 prefill + decode", "短 prompt(10 < 32) 不触发 chunk",
         [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
         _make_sampling_params(SamplingParams, max_tokens=4)),

        ("多条 prefill", "4 个短 prompt 同批 prefill",
         [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
         _make_sampling_params(SamplingParams, max_tokens=3)),

        ("chunked prefill → prefix-prefill", "60 token 长 prompt，切成 32+28 两块",
         [list(range(1, 61))],
         _make_sampling_params(SamplingParams, max_tokens=2)),
    ]

    # ---- 4. 逐个阶段跑 ----
    print("\n" + "=" * 70)
    print(f"共 {len(stages)} 个阶段测试，逐一执行 ...")
    for name, desc, prompts, sampling_params in stages:
        print("\n" + "-" * 70)
        print(f"[阶段] {name}")
        print(f"       说明: {desc}   (batch={len(prompts)}, max_tokens={sampling_params.max_tokens})")
        print("-" * 70)
        outputs = llm.generate(prompts, sampling_params)
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            gen = output.outputs[0]
            print(f"  prompt[{i}] 输入 {len(prompt)} token → "
                  f"生成 {len(gen.token_ids)} token: {gen.token_ids}")

    print("\n" + "=" * 70)
    print("全部阶段测试执行完毕")


if __name__ == "__main__":
    main()

