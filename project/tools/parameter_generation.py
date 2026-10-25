import os
import json
import torch
import torch.nn as nn

def generate_and_save_parameters():
    # 1. 确保目标文件夹 project/data/parameter/ 存在（唯一真源）
    save_dir = "/home/xkem/cuda_triton/project/data/parameter"
    os.makedirs(save_dir, exist_ok=True)
    
    config_path = os.path.join(save_dir, "model_config.json")
    weights_path = os.path.join(save_dir, "model_weights.pt")

    # =====================================================================
    # 2. 定义超参数配置 (JSON)
    # =====================================================================
    config = {
        "d_model": 512,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "num_experts": 8,
        "top_k": 2,
        "d_ffn": 1024,
        # ---- 折叠参数（token id 除以几，供 SELF-LLM.py / 主程序共用）----
        "vocab_full": 151936,   # Qwen config 口径（折叠基数）
        "fold_ratio": 10,       # 官方 BPE token id // 10
    }

    # 保存 JSON 配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"✅ 配置文件已成功写入: {config_path}")

    # =====================================================================
    # 3. 构造对应的权重字典 (State Dict)
    # =====================================================================
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    num_experts = config["num_experts"]
    d_ffn = config["d_ffn"]

    # 折叠后词表行数 V = ceil(vocab_full / fold_ratio)
    vocab_folded = (config["vocab_full"] + config["fold_ratio"] - 1) // config["fold_ratio"]

    state_dict = {}

    # --- 第二层 GroupTransformer 的权重（组合后 key 带 transformer. 前缀） ---
    # RMSNorm 层的 gamma 权重初始化为全 1
    state_dict["transformer.input_norm.weight"] = torch.ones(d_model, dtype=torch.float16)
    state_dict["transformer.post_attn_norm.weight"] = torch.ones(d_model, dtype=torch.float16)

    # Attention 投影矩阵（标准正态分布缩放初始化，全链路 fp16）
    state_dict["transformer.q_proj.weight"] = torch.randn(num_heads * head_dim, d_model, dtype=torch.float16) * 0.02
    state_dict["transformer.k_proj.weight"] = torch.randn(num_kv_heads * head_dim, d_model, dtype=torch.float16) * 0.02
    state_dict["transformer.v_proj.weight"] = torch.randn(num_kv_heads * head_dim, d_model, dtype=torch.float16) * 0.02
    state_dict["transformer.out_proj.weight"] = torch.randn(d_model, num_heads * head_dim, dtype=torch.float16) * 0.02

    # --- MoEInferenceLayer 层的权重 ---
    # Router 路由权重：采用极小方差的正态分布初始化，确保概率分布均衡，避免特定专家过载
    state_dict["transformer.moe_layer.router.weight"] = torch.randn(num_experts, d_model, dtype=torch.float16) * 0.001

    # 专家权重矩阵：满足 Triton TMA Grouped GEMM 的物理内存布局
    # W1 (Up-proj): [num_experts, d_ffn, d_model]
    # W2 (Down-proj): [num_experts, d_model, d_ffn]
    state_dict["transformer.moe_layer.w1"] = torch.randn(num_experts, d_ffn, d_model, dtype=torch.float16) * 0.02
    state_dict["transformer.moe_layer.w2"] = torch.randn(num_experts, d_model, d_ffn, dtype=torch.float16) * 0.02

    # --- 第一层 embedding 端权重 ---
    # 折叠后词表 -> d_model 维，fp16 随机；key 必须与 SelfLLM.embed_tokens 严格一致
    state_dict["embed_tokens.weight"] = torch.randn(vocab_folded, d_model, dtype=torch.float16) * 0.02

    # --- 第三层 预测层权重 ---
    # d_model -> vocab_folded，fp16 随机；key 必须与 SelfLLM.lm_head 严格一致
    state_dict["lm_head.weight"] = torch.randn(vocab_folded, d_model, dtype=torch.float16) * 0.02

    # 保存二进制权重文件 (.pt)
    torch.save(state_dict, weights_path)
    print(f"✅ 权重文件已成功写入: {weights_path}")
    print(f"   权重总项数: {len(state_dict)}（含 embed_tokens.weight [{vocab_folded}, {d_model}]）")
    print("   全链路数据类型: torch.float16")

if __name__ == "__main__":
    generate_and_save_parameters()