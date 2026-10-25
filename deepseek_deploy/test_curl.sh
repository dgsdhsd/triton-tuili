#!/usr/bin/env bash
# =============================================================================
# 用 curl 直接测 vLLM 的 OpenAI 兼容 HTTP 接口（底层，便于看清请求格式）
# 前置：先跑 serve.sh / serve_single.sh 把服务拉起来
# =============================================================================
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000/v1}"

echo "==> 1) 模型列表  GET /v1/models"
curl -s "$BASE/models"
echo

echo "==> 2) 补全接口  POST /v1/completions  （Base 模型用这个：代码补全）"
curl -s "$BASE/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "prompt": "def quick_sort(arr):\n",
    "max_tokens": 64,
    "temperature": 0.2,
    "stop": ["\n\n"]
  }'
echo

echo "==> 3) 对话接口  POST /v1/chat/completions  （Instruct 模型用这个：多轮对话）"
curl -s "$BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "messages": [{"role": "user", "content": "用一句话解释什么是张量并行？"}],
    "max_tokens": 64,
    "temperature": 0.2
  }'
echo
