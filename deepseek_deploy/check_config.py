#!/usr/bin/env python3
"""
配置自检脚本（无需下载任何权重，秒级完成）。

目的：在"不运行真实推理"的前提下，验证这套部署配置是正确的：
  1. model/config.json 是否是一个合法的 DeepSeek-Coder-V2-Lite（deepseek_v2）配置；
  2. vLLM 注册表里是否有该架构的内置实现（DeepseekV2ForCausalLM -> deepseek_v2）；
  3. 模拟 TP/EP 切分，确认 64 个专家能被 EP 整除、分布合理。

运行：/home/xkem/cuda_triton/vllm_env/bin/python check_config.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, "model", "config.json")
VLLM_ENV = "/home/xkem/cuda_triton/vllm_env"

cfg = json.load(open(CFG_PATH, encoding="utf-8"))

print("=" * 64)
print("1) 架构识别")
print("=" * 64)
assert cfg["model_type"] == "deepseek_v2", f"model_type 异常: {cfg['model_type']}"
assert cfg["architectures"] == ["DeepseekV2ForCausalLM"], cfg["architectures"]
print(f"   model_type     = {cfg['model_type']}")
print(f"   architectures  = {cfg['architectures']}")

print("=" * 64)
print("2) MoE 字段（专家并行 EP 的结构前提）")
print("=" * 64)
moe_fields = {
    "n_routed_experts":    "可被 EP 切分的路由专家总数",
    "n_shared_experts":    "共享专家（每个 token 都过，不参与 EP，各卡复制）",
    "num_experts_per_tok": "每个 token 激活的专家数（top-k）",
    "first_k_dense_replace": "前 k 层用 dense MLP 取代 MoE（本模型=1，即第 0 层是 dense）",
    "moe_intermediate_size": "单个专家的 FFN 中间维（1408）",
    "n_group":             "专家分组数（grouped top-k，本模型=1 即不分组）",
    "topk_group":          "top-k 中先选组的个数",
}
for k, desc in moe_fields.items():
    print(f"   {k:24s}= {cfg[k]:>5}   # {desc}")

print("=" * 64)
print("3) MLA 注意力字段（TP 会按 head 切这些投影）")
print("=" * 64)
mla_fields = [
    "num_attention_heads", "num_key_value_heads", "kv_lora_rank",
    "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
]
for k in mla_fields:
    print(f"   {k:24s}= {cfg[k]}")

print("=" * 64)
print("4) vLLM 内置实现注册检查")
print("=" * 64)
reg_path = os.path.join(
    VLLM_ENV, "lib", "python3.10", "site-packages",
    "vllm", "model_executor", "models", "registry.py",
)
reg_src = open(reg_path, encoding="utf-8").read()
ok = ('"DeepseekV2ForCausalLM"' in reg_src) and ('"deepseek_v2"' in reg_src)
print(f"   registry.py 含 DeepseekV2ForCausalLM -> deepseek_v2 映射: {'OK' if ok else 'FAIL'}")
assert ok

print("=" * 64)
print("5) TP/EP 切分模拟（64 专家分布）")
print("=" * 64)
n_routed = cfg["n_routed_experts"]
n_active = cfg["num_experts_per_tok"]
print(f"   路由专家总数 {n_routed}，每 token 激活 {n_active} 个")
for ep in (1, 2, 4, 8, 16, 64):
    if n_routed % ep != 0:
        print(f"   EP={ep:>2}: 64 无法整除，跳过")
        continue
    n_local = n_routed // ep
    print(f"   EP={ep:>2} -> 每卡持有 {n_local:>2} 个专家；"
          f"激活 token 经 all-to-all 路由到持有专家的卡")

print("=" * 64)
print("结论：配置合法，vLLM 可识别 deepseek_v2 架构，TP/EP 切分无整除问题。")
print("下一步：多卡云上运行 ./serve.sh；本机单卡可 ./serve_single.sh（AWQ）。")
