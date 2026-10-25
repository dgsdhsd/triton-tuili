#!/usr/bin/env python3
"""
用 OpenAI Python SDK 测 vLLM 的 OpenAI 兼容接口。

前置：先启动 serve.sh / serve_single.sh。
运行：/home/xkem/cuda_triton/vllm_env/bin/python test_api.py

说明：
  - /v1/completions   给 Base 模型（代码补全，无 chat template）
  - /v1/chat/completions 给 Instruct 模型（多轮对话）
  Base 模型没有对话模板，chat 接口可能报错或退化，脚本会捕获并提示。
"""
import os

from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-coder-v2-lite")

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def test_completions():
    """Base 模型：代码补全接口。"""
    print("=" * 60)
    print("[1] POST /v1/completions  (代码补全)")
    print("=" * 60)
    resp = client.completions.create(
        model=MODEL,
        prompt="def fibonacci(n):\n    \"\"\"返回斐波那契数列第 n 项\"\"\"\n",
        max_tokens=128,
        temperature=0.2,
        stop=["\n\n"],
    )
    text = resp.choices[0].text
    print(f"model={resp.model}, finish={resp.choices[0].finish_reason}")
    print(f"补全结果:\n{text}")


def test_chat():
    """Instruct 模型：多轮对话接口。Base 模型无 chat template，可能失败。"""
    print("=" * 60)
    print("[2] POST /v1/chat/completions  (多轮对话)")
    print("=" * 60)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是代码助手。"},
                {"role": "user", "content": "用一句话解释什么是张量并行？"},
            ],
            max_tokens=128,
            temperature=0.2,
        )
        print(f"model={resp.model}, finish={resp.choices[0].finish_reason}")
        print(f"回复:\n{resp.choices[0].message.content}")
    except Exception as e:  # noqa: BLE001
        print(f"[提示] chat 接口调用失败（Base 模型无 chat template 属正常现象）: {e}")
        print("       若要真正对话，请改用 DeepSeek-Coder-V2-Lite-Instruct。")


def main():
    test_completions()
    print()
    test_chat()


if __name__ == "__main__":
    main()
