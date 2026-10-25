#!/usr/bin/env python3
"""
SELF-LLM 纯 torch 端到端主程序（模型接口的"另一个程序"入口）。

流程：
    文本 → 官方 BPE 分词 → //fold_ratio → 查表(embedding) → GroupTransformer → hidden
    （预测端 logits / 采样 / ×10 / decode 本轮为占位演示，见代码 TODO）

运行：
    /home/xkem/cuda_triton/vllm_env/bin/python project/self_llm_pipeline.py
"""
import importlib.util
import sys

import torch


def load_self_llm_module():
    """以 importlib 加载 SELF-LLM.py（文件名带连字符，无法直接 import）。"""
    path = "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
    spec = importlib.util.spec_from_file_location("self_llm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["self_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    self_llm = load_self_llm_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 读配置 + 建模型 + 灌权重（strict，key 完全对齐）
    config = self_llm.load_model_config(self_llm.CONFIG_PATH)
    model = self_llm.build_self_llm(config).to(device)
    self_llm.load_model_weights(model, self_llm.WEIGHTS_PATH)
    model.eval()

    fold_ratio = config["fold_ratio"]
    print(f"[配置] vocab_full={config['vocab_full']} fold_ratio={fold_ratio} "
          f"vocab_folded={model.vocab_folded} d_model={config['d_model']}")

    # 2. 官方 BPE + 折叠（独立于模型）
    tokenizer_dir = (
        "/home/xkem/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    bpe = self_llm.QwenBPE(tokenizer_dir, fold_ratio=fold_ratio)

    # 3. 文本 → 大 id → 折叠 → 小 id
    text = "今天天气很好，适合写一个 MoE 大语言模型。"
    big_ids = bpe.encode(text)
    small_ids = bpe.encode_folded(text)
    print(f"[BPE] 文本: {text}")
    print(f"[BPE] 大 id（{len(big_ids)} 个）: {big_ids}")
    print(f"[BPE] 小 id（{len(small_ids)} 个，//{fold_ratio}）: {small_ids}")

    # 4. 完整前向：embedding → transformer → 预测层（三层一次走完，输出 logits）
    input_ids = torch.tensor(small_ids, dtype=torch.long, device=device)
    cu_seqlens = torch.tensor([0, len(small_ids)], dtype=torch.int32, device=device)
    with torch.no_grad():
        logits = model(input_ids, cu_seqlens, len(small_ids))
    print(f"[模型] embedding + transformer + lm_head 输出 logits: {tuple(logits.shape)}")

    # 5. 采样（argmax 演示）→ decode，走完"预测 → 文本"这最后一环
    small_out = logits.argmax(dim=-1)  # [T] 小 id
    text_out = bpe.decode_folded(small_out.tolist())
    print(f"[采样] argmax 小 id（{len(small_out)} 个）: {small_out.tolist()}")
    print(f"[输出] decode_folded → 文本（乱码，预期）: {text_out!r}")

    # 6. 折叠往返演示（有损映射，属预期）
    text_back = bpe.decode_folded(small_ids)
    print(f"[往返] decode_folded（有损，预期乱码）: {text_back!r}")
    print(f"[往返] 示例: 原 id={big_ids[0]} → 折叠={small_ids[0]} → 还原={small_ids[0] * fold_ratio}")


if __name__ == "__main__":
    main()
