#!/usr/bin/env bash
# Chapter 11 (推奨): mlx_lm.server で「本体モデル + LoRAアダプタ」をそのまま配信する。
# GGUF変換が要らず最も確実。OpenAI互換APIが http://localhost:8080 に立つ。
#
#   bash scripts/08_serve.sh
#
# 別ターミナルで:  uv run python scripts/08_client.py "領収書を再発行したい"
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${COURSE_MODEL:-mlx-community/gemma-3n-E2B-it-lm-4bit}"
echo "serving: $MODEL  + adapters/  on :8080"
uv run mlx_lm.server --model "$MODEL" --adapter-path adapters --port 8080
