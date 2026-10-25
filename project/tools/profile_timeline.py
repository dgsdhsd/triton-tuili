"""用 torch.profiler(CUPTI) 抓取 GroupTransformer 前向的 GPU kernel 时间轴。

背景：本机 nsys 2024.6.2 自带的 CUPTI 太旧，无法在 RTX 5070 (Blackwell sm_120)
上采集 GPU kernel 轨迹；但系统 CUDA 12.8 自带的 CUPTI (2025.1.1) 可以正常工作。
因此这里用 torch.profiler 生成等价的 Chrome trace 时间轴 + kernel 汇总表。
"""
import sys
import importlib.util

import torch
from torch.profiler import profile, ProfilerActivity, record_function

# 复用 MOE-LLM.py 里已定义好的模型类与加载函数，避免重复实现
MOE_PATH = "/home/xkem/cuda_triton/project/MOE-LLM.py"
spec = importlib.util.spec_from_file_location("moe_llm", MOE_PATH)
moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe)

device = "cuda" if torch.cuda.is_available() else "cpu"

config_file_path = "/home/xkem/cuda_triton/project/parameter/model_config.json"
weights_file_path = "/home/xkem/cuda_triton/project/parameter/model_weights.pt"

model_kwargs = moe.load_model_config(config_file_path)
model = moe.GroupTransformer(**model_kwargs).to(device)
moe.load_model_weights(model, weights_file_path)

seqlens = [60, 100, 250, 480, 950, 2000]
cu_seqlens = torch.tensor([0, 60, 160, 410, 890, 1840, 3840], dtype=torch.int32, device=device)
max_seqlen = max(seqlens)
total_tokens = cu_seqlens[-1].item()
x = torch.randn(total_tokens, model_kwargs["d_model"], dtype=torch.float16, device=device)

model.eval()
with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("GroupTransformer.forward"):
            out = model(x, cu_seqlens, max_seqlen)
        torch.cuda.synchronize()

print("=" * 100)
print("GPU kernel 时间汇总 (按 CUDA 时间排序, Top 60):")
print("=" * 100)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))

trace_path = "/home/xkem/cuda_triton/project/moe_llm_timeline.json"
prof.export_chrome_trace(trace_path)
print(f"\n✅ Chrome trace 已导出: {trace_path}")
print("   在浏览器打开 chrome://tracing 并 Load 该文件，即可看到完整计算时间轴图。")
print("   (也可用 https://ui.perfetto.dev 打开)")
print(f"输出 Tensor 形状: {out.shape}")
