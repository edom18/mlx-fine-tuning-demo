#!/usr/bin/env bash
# Chapter 11.5 (Unity / llama.cpp 直系): LoRAを本体へ融合し、llama.cpp の GGUF へ変換して
# 「LLM for Unity」(内部 llama.cpp) で使える 1ファイルの量子化GGUF を作る。
#
#   bash scripts/10_to_gguf.sh
#
# 出力: fused_model/support-classifier-<QUANT>.gguf  ← これを LLMUnity にカスタムモデルとして指定。
#       LoRA は焼き込み済みなので AddLora は不要 (Route A = 融合1枚)。
#
# 環境変数で上書き可:
#   COURSE_MODEL   ベースモデル   (default: mlx-community/gemma-3n-E2B-it-lm-4bit)
#   ADAPTER_PATH   学習済みアダプタ (default: adapters)
#   LLAMACPP_DIR   llama.cpp の場所 (default: $HOME/llama.cpp、無ければ clone)
#   QUANT          量子化タイプ    (default: Q4_K_M)
#   FORCE=1        既存生成物を無視して各段をやり直す
#   SKIP_SMOKE=1   最後のスモークテストを省略
#   SMOKE_INPUT    スモークテストの入力文
#
# 実機監査(2026-06, mlx-lm 0.31.3 / llama.cpp master):
#   - gemma-3n は llama.cpp(master) でテキスト変換に対応済み (conversion/gemma.py)。
#   - mlx_lm.fuse のフラグは --dequantize (1語) で、4bitのままだと変換でこけるため必須。
#   - convert_hf_to_gguf.py は torch を import するので venv に torch が要る。
#   - スモークは llama-cli ではなく llama-completion (llama-cli は対話専用化)。
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${COURSE_MODEL:-mlx-community/gemma-3n-E2B-it-lm-4bit}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
QUANT="${QUANT:-Q4_K_M}"
FORCE="${FORCE:-}"
FUSED="fused_model"
OUT_F16="$FUSED/support-classifier-f16.gguf"
OUT_Q="$FUSED/support-classifier-${QUANT}.gguf"

echo "============================================================"
echo " MLX LoRA → GGUF  (Unity / llama.cpp 用)"
echo "   model   : $MODEL"
echo "   adapter : $ADAPTER_PATH"
echo "   llama   : $LLAMACPP_DIR"
echo "   output  : $OUT_Q"
echo "============================================================"

# ---------- [1/5] アダプタを本体へ融合 (dequantize 必須) ----------
if [[ -n "$FORCE" || ! -f "$FUSED/config.json" ]]; then
  echo "[1/5] fuse (--dequantize): $ADAPTER_PATH → $FUSED/"
  uv run mlx_lm.fuse --model "$MODEL" --adapter-path "$ADAPTER_PATH" \
    --save-path "$FUSED" --dequantize
else
  echo "[1/5] fuse 済み ($FUSED/config.json あり) — skip (FORCE=1 で再実行)"
fi

# ---------- [1.5] SPM tokenizer.model を補完 (Unityの古いllama.cpp対策) ----------
# mlx_lm.fuse は tokenizer.model(SPM) を fused_model にコピーしない。これが無いと
# convert_hf_to_gguf が BPE(gpt2)経路に落ち、新しい pre-tokenizer 名が付与され、
# 古いビルド(LLM for Unity同梱等)で 'unknown pre-tokenizer type' で読めなくなる。
# SPM を置くと SPM経路になり tokenizer.ggml.model=llama / pre=default で広く互換になる。
if [[ ! -f "$FUSED/tokenizer.model" ]]; then
  CACHE="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots"
  SRC=$(ls -d "$CACHE"/*/ 2>/dev/null | head -1)
  if [[ -n "$SRC" && -f "${SRC}tokenizer.model" ]]; then
    cp "${SRC}tokenizer.model" "$FUSED/"
    [[ -f "${SRC}special_tokens_map.json" ]] && cp "${SRC}special_tokens_map.json" "$FUSED/"
    rm -f "$OUT_F16" "$OUT_Q"   # 旧(BPE)GGUFがあればSPMで作り直す
    echo "[1.5] SPM tokenizer.model を補完 (${SRC} → $FUSED/)。既存GGUFは作り直します"
  else
    echo "[1.5] 警告: tokenizer.model が見つからず ($MODEL)。BPE経路となり古いllama.cppで読めない場合あり"
  fi
else
  echo "[1.5] tokenizer.model 既にあり — skip"
fi

# ---------- [2/5] llama.cpp と torch を用意 ----------
if [[ ! -f "$LLAMACPP_DIR/convert_hf_to_gguf.py" ]]; then
  echo "[2/5] llama.cpp を clone → $LLAMACPP_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
else
  echo "[2/5] llama.cpp あり: $LLAMACPP_DIR"
fi
if ! uv run python -c "import torch" 2>/dev/null; then
  echo "      convert_hf_to_gguf.py 用に torch を追加 (uv pip install torch)"
  uv pip install torch
fi

# ---------- [3/5] HF → GGUF (f16) ----------
if [[ -n "$FORCE" || ! -f "$OUT_F16" ]]; then
  echo "[3/5] convert_hf_to_gguf: $FUSED → $OUT_F16"
  uv run python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$FUSED" \
    --outfile "$OUT_F16" --outtype f16
else
  echo "[3/5] $OUT_F16 あり — skip"
fi

# ---------- [4/5] ビルド & 量子化 ----------
QUANT_BIN="$LLAMACPP_DIR/build/bin/llama-quantize"
COMPL_BIN="$LLAMACPP_DIR/build/bin/llama-completion"
if [[ ! -x "$QUANT_BIN" || ! -x "$COMPL_BIN" ]]; then
  echo "      llama.cpp をビルド (Metal: llama-quantize / llama-completion)"
  cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
  cmake --build "$LLAMACPP_DIR/build" -j --target llama-quantize llama-completion
fi
if [[ -n "$FORCE" || ! -f "$OUT_Q" ]]; then
  echo "[4/5] quantize → $OUT_Q ($QUANT)"
  "$QUANT_BIN" "$OUT_F16" "$OUT_Q" "$QUANT"
else
  echo "[4/5] $OUT_Q あり — skip"
fi

# ---------- [5/5] スモークテスト (llama-completion) ----------
if [[ -z "${SKIP_SMOKE:-}" ]]; then
  echo "[5/5] スモークテスト:"
  Q="${SMOKE_INPUT:-ログインできません。パスワードを再設定したいです。}"
  uv run python -c "
import sys; sys.path.insert(0,'scripts')
import common
print('<start_of_turn>user'); print(common.system_prompt()); print('')
print(sys.argv[1] + '<end_of_turn>'); print('<start_of_turn>model')
" "$Q" > /tmp/10_to_gguf_prompt.txt
  echo "  IN : $Q"
  printf "  OUT:"
  "$COMPL_BIN" -m "$OUT_Q" -f /tmp/10_to_gguf_prompt.txt \
    -n 12 --temp 0 -no-cnv -ngl 99 --no-display-prompt 2>/dev/null || true
  echo
else
  echo "[5/5] スモークテスト skip (SKIP_SMOKE=1)"
fi

echo "============================================================"
echo " 完成: $OUT_Q"
echo " → 「LLM for Unity」にカスタムモデルとして指定すれば動きます。"
echo "    (LoRA は焼き込み済みなので AddLora は不要)"
echo "============================================================"
