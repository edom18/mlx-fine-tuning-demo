#!/usr/bin/env bash
# Chapter 11 (Ollamaに持ち込みたい人向け): アダプタを本体に融合し、GGUFへ変換して Ollama に登録する。
#
# 注意: GGUFエクスポートは「対応アーキテクチャ」のみ。Gemma 3n は MatFormer 等の新機構があり、
#       mlx/llama.cpp/Ollama 側の対応状況によっては変換に失敗することがある。その場合は
#       scripts/08_serve.sh の mlx_lm.server 配信を使うのが確実(機能は同じ＝ローカルでLoRA済みモデルを運用できる)。
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${COURSE_MODEL:-mlx-community/gemma-3n-E2B-it-lm-4bit}"

echo "[1/4] アダプタを本体へ融合 (adapters/ → fused_model/)"
uv run mlx_lm.fuse --model "$MODEL" --adapter-path adapters --save-path fused_model

echo "[2/4] GGUF へエクスポートを試みる (fused_model/ → fused_model/ggml-model-f16.gguf)"
# 方法A: mlx 側で直接エクスポート(対応アーキのみ)
if uv run mlx_lm.fuse --model "$MODEL" --adapter-path adapters \
      --export-gguf --gguf-path fused_model/ggml-model-f16.gguf ; then
  echo "  → mlx での GGUF エクスポート成功"
else
  echo "  → mlx での GGUF エクスポート不可。llama.cpp の convert を使うか、08_serve.sh に切替を検討。"
  echo "     例) python /path/to/llama.cpp/convert_hf_to_gguf.py fused_model --outfile fused_model/model.gguf"
fi

echo "[3/4] Ollama へ登録 (Modelfile を参照)"
echo "      Modelfile 内の FROM のパスを実際の .gguf に合わせてから:"
echo "      ollama create support-classifier -f Modelfile"

echo "[4/4] 実行"
echo "      ollama run support-classifier '領収書を再発行したいです'"
