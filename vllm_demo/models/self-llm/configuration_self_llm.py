"""SELF-LLM 的 HF 风格配置类（配合 config.json 的 model_type="self_llm"）。"""

from transformers import PretrainedConfig


class SelfLLMConfig(PretrainedConfig):
    model_type = "self_llm"

    def __init__(
        self,
        vocab_size: int = 15194,
        hidden_size: int = 512,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        num_experts: int = 8,
        top_k: int = 2,
        d_ffn: int = 1024,
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
