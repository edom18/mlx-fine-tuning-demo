"""
Chapter 4 ハンズオン: 素の Gemma 3n E2B が、ファインチューニング無しで
どこまで問い合わせ分類できるかを測る（=これがベースライン）。
ここで「ラベル外の余計な文章を返す」「迷うと説明し始める」などの弱点を体感する。

実行:
    uv run python scripts/02_baseline_eval.py
"""
from __future__ import annotations
from mlx_lm import load
import common
import classifier


def main() -> None:
    rows = common.read_jsonl(common.DATA / "seed.jsonl")
    print(f"ベースライン評価: 素のモデル {common.model_id()} / {len(rows)}件")
    model, tokenizer = load(common.model_id())
    metrics = classifier.evaluate(model, tokenizer, rows)
    classifier.print_report(metrics, title="ベースライン（ファインチューニング前）")
    print("\nこの数字を覚えておく。Chapter 10 でファインチューニング後と比較する。")


if __name__ == "__main__":
    main()
