import os
import json
import tempfile
import importlib.util
import sys

# ---- 环境变量护栏（必须在 vLLM 引擎初始化前设置，WSL2 + RTX 5070/Blackwell）----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 关闭多进程，让自定义模型注册在引擎进程内生效
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # 5070 上 FlashInfer 采样器有 bug
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")  # WSL2 下启用 pinned memory（否则 UVA 不可用）

# ---- vLLM 相关依赖 ----
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry


# =====================================================================
# 1. 加载 SELF-LLM.py（文件名带连字符，无法直接 import，用 importlib）
# =====================================================================
def load_self_llm_module():
    path = "/home/xkem/cuda_triton/project/models/SELF-LLM.py"
    spec = importlib.util.spec_from_file_location("self_llm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["self_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# 2. 动态生成一个最小的 HF 远程配置类，骗过 transformers 的 AutoConfig
# =====================================================================
SELF_LLM_CONFIG_PY = '''\
"""SelfLLM 的 HF 风格配置类（配合 config.json 的 model_type=\"self_llm\"）。"""

from transformers import PretrainedConfig


class SelfLLMConfig(PretrainedConfig):
    model_type = "self_llm"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 512,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        num_experts: int = 8,
        top_k: int = 2,
        d_ffn: int = 2048,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ffn = d_ffn
        self.tie_word_embeddings = tie_word_embeddings
'''


# =====================================================================
# 3. 主程序：注册 SELF-LLM.py 的 SelfLLM 并跑通 prefill + decode
# =====================================================================
def main():
    print("[1] 加载 SELF-LLM.py 并注册 SelfLLM 到 vLLM ...")
    self_llm = load_self_llm_module()
    ModelRegistry.register_model("SelfLLM", self_llm.SelfLLM)

    dummy_config = {
        "model_type": "self_llm",
        "architectures": ["SelfLLM"],  # 必须和 register_model 时的名字一致
        "auto_map": {"AutoConfig": "configuration_self_llm.SelfLLMConfig"},
        "vocab_size": 151936,
        "hidden_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_experts": 8,
        "top_k": 2,
        "d_ffn": 2048,
        "torch_dtype": "float16",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dummy_config, f)

        with open(os.path.join(temp_dir, "configuration_self_llm.py"), "w") as f:
            f.write(SELF_LLM_CONFIG_PY)

        print(f"[2] 启动 vLLM 引擎 (假权重模式), 配置加载自: {temp_dir}")
        llm = LLM(
            model=temp_dir,
            load_format="dummy",          # 不加载真实权重，避免报错
            skip_tokenizer_init=True,     # 无 tokenizer，直接喂 token ids
            trust_remote_code=True,       # 加载 configuration_self_llm.py
            tensor_parallel_size=1,
            gpu_memory_utilization=0.20,
            max_model_len=128,
            enforce_eager=True,           # 关闭 CUDA Graph
            dtype="float16",
        )

        # 直接喂 token ids（必须 < vocab_size）
        prompts = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # 10 个 token -> PREFILL 一次吃 10
            [100, 200, 300],                    # 3 个 token
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=3,        # 只生成 3 个 token，看 3 次 Decode
            ignore_eos=True,     # 强制生成满 3 个
            detokenize=False,    # 直接输出 token ids
        )

        print("\n[3] 🚀 开始推理 (1 次 Prefill + 3 次 Decode) ...")
        outputs = llm.generate(prompts, sampling_params)

        print("\n[4] ✅ 推理完成！成功调用 SELF-LLM.py 的 SelfLLM，最终结果：")
        for i, out in enumerate(outputs):
            gen = out.outputs[0]
            print(f"  请求 {i}: PromptIds = {out.prompt_token_ids} | 生成Ids = {gen.token_ids}")


if __name__ == "__main__":
    main()
