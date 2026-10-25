yimport os
import json
import tempfile
import torch
import torch.nn as nn
from typing import Any, List

from vllm import LLM, SamplingParams

# 动态适配 ModelRegistry 导入路径 (兼容老版本和新版本)
try:
    from vllm.model_executor.models.registry import ModelRegistry
except ImportError:
    from vllm.model_executor.models import ModelRegistry

# 动态适配 LogitsProcessor 和 Sampler
try:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.sampler import Sampler
except ImportError:
    raise ImportError("无法导入 LogitsProcessor 或 Sampler，请检查 vLLM 核心库安装状态。")


# =====================================================================
# 1. 核心模型定义：SelfLLM 与 动态探针 MockGroupTransformer
# =====================================================================

class MockGroupTransformer(nn.Module):
    """
    用作“动态探针”，拦截并打印出 vLLM 在你当前版本下究竟传了哪些 Tensor 和参数。
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = getattr(config, "hidden_size", 512)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, 
                kv_caches: List[torch.Tensor], attn_metadata: Any, **kwargs):
        print("\n" + "="*65)
        
        # 启发式判断运行阶段（兼容 prefill_tokens / is_prompt / num_prefills 等各版本命名）
        is_prefill = False
        if hasattr(attn_metadata, "num_prefill_tokens") and getattr(attn_metadata, "num_prefill_tokens", 0) > 0:
            is_prefill = True
        elif hasattr(attn_metadata, "is_prompt") and getattr(attn_metadata, "is_prompt", False):
            is_prefill = True
        elif hasattr(attn_metadata, "num_prefills") and getattr(attn_metadata, "num_prefills", 0) > 0:
            is_prefill = True
            
        if is_prefill:
            print("🚀 [运行阶段] PREFILL (全量前向传播，处理变长 Ragged Tensor)")
        else:
            print("⚡ [运行阶段] DECODE (自回归生成，基于 PagedAttention 查表)")
            
        print(f"  -> 输入 Hidden States 尺寸: {list(hidden_states.shape)}")
        print(f"  -> 位置编码 Positions 尺寸  : {list(positions.shape)}")
        
        print("  -> 【核心揭秘】当前 vLLM 版本 attn_metadata 内部包含的所有有效变量:")
        # 动态遍历 metadata，避免版本差异导致找错属性名
        for attr_name in dir(attn_metadata):
            if attr_name.startswith("_") or attr_name in ["device", "dtype"]:
                continue
            val = getattr(attn_metadata, attr_name)
            if isinstance(val, torch.Tensor):
                print(f"     * {attr_name: <22}: Tensor of shape {list(val.shape)} (dtype: {val.dtype})")
            elif isinstance(val, (int, float, bool, list)):
                print(f"     * {attr_name: <22}: {val}")

        if kv_caches and kv_caches[0] is not None:
            print(f"  -> 物理 KV Cache 分配尺寸 : {list(kv_caches[0].shape)}")
        else:
            print("  -> 物理 KV Cache 状态     : 未分配或为空")

        print("="*65)
        
        # 原样返回假装计算完毕
        return hidden_states


class SelfLLM(nn.Module):
    """
    适配 vLLM 的顶层模型类。加入了 args 兜底机制以免疫接口变化。
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # 兼容各版本 vLLM 初始化 config 的传参方式
        if "vllm_config" in kwargs:
            hf_config = kwargs["vllm_config"].model_config.hf_config
        elif len(args) > 0 and hasattr(args[0], "model_config"):
            hf_config = args[0].model_config.hf_config
        else:
            hf_config = args[0] if len(args) > 0 else kwargs.get("config")
            
        self.vocab_size = getattr(hf_config, "vocab_size", 151936)
        self.d_model = getattr(hf_config, "hidden_size", 512)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model, dtype=torch.float16)
        self.transformer = MockGroupTransformer(hf_config)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=torch.float16)

        # 初始化必需的处理器
        self.logits_processor = LogitsProcessor(vocab_size=self.vocab_size)
        self.sampler = Sampler()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args, **kwargs):
        """
        利用 *args 和 **kwargs 实现高鲁棒性的传参捕获。
        """
        # 动态捕捉变幻莫测的传参位置
        kv_caches = kwargs.get("kv_caches") if "kv_caches" in kwargs else (args[0] if len(args) > 0 else None)
        attn_metadata = kwargs.get("attn_metadata") if "attn_metadata" in kwargs else (args[1] if len(args) > 1 else None)

        x = self.embed_tokens(input_ids)
        hidden = self.transformer(x, positions, kv_caches, attn_metadata, **kwargs)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata, **kwargs):
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata, **kwargs)

    def sample(self, logits: torch.Tensor, sampling_metadata, **kwargs):
        return self.sampler(logits, sampling_metadata, **kwargs)

    def load_weights(self, weights):
        """Dummy模式下无需真实加载逻辑，占位防报错"""
        pass


# =====================================================================
# 2. 最小化独立运行入口
# =====================================================================
def main():
    # 强制单卡运行
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("[1] 注册自定义模型 SelfLLM 到 vLLM ...")
    ModelRegistry.register_model("SelfLLM", SelfLLM)

    # 动态创建架构配置
    dummy_config = {
        "architectures": ["SelfLLM"], 
        "hidden_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "head_dim": 64
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dummy_config, f)

        print(f"[2] 启动 vLLM 引擎, 读取临时配置: {temp_dir}")
        llm = LLM(
            model=temp_dir,
            load_format="dummy",          
            tensor_parallel_size=1,
            gpu_memory_utilization=0.20, 
            max_model_len=128,
            enforce_eager=True,           # 必须关闭 CUDA Graph，否则 print 打印不出来！
            dtype="float16"
        )

        prompts = [
            "Hello, this is a test.",
            "GroupTransformer is a new custom architecture that we are testing."
        ]
        
        sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=3, # 强制执行 3 次 Decode 阶段
        )

        print("\n[3] 🚀 开始推理执行阶段 ...")
        outputs = llm.generate(prompts, sampling_params)

        print("\n[4] 运行结束！")

if __name__ == "__main__":
    main()