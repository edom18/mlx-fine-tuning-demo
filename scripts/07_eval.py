"""
Chapter 10 ハンズオン: 評価設計。test.jsonl(人手・リーク無し)で
ファインチューニング前後を同じ物差しで比較する。

指標: accuracy / ラベル外出力率 / ラベル別 precision・recall・f1 / 混同行列 / 誤判定例

実行:
    uv run python scripts/07_eval.py                 # base と fine-tuned を両方測って比較
    uv run python scripts/07_eval.py --only base     # 素のモデルだけ
    uv run python scripts/07_eval.py --only ft --adapter-path adapters
"""
from __future__ import annotations
import argparse
from mlx_lm import load
import common
import classifier


def load_test(path) -> list[dict]:
    """chat形式 test.jsonl → {text,label} へ。"""
    rows = []
    for r in common.read_jsonl(path):
        msgs = r["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        rows.append({"text": user, "label": gold})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA / "test.jsonl"))
    ap.add_argument("--adapter-path", default="adapters")
    ap.add_argument("--only", choices=["base", "ft"], default=None)
    args = ap.parse_args()

    rows = load_test(args.data)
    print(f"評価データ: {args.data}  ({len(rows)}件)\n")

    base_acc = ft_acc = None
    if args.only != "ft":
        model, tok = load(common.model_id())
        m = classifier.evaluate(model, tok, rows)
        classifier.print_report(m, title="① ファインチューニング前 (base)")
        base_acc = m["accuracy"]

    if args.only != "base":
        model, tok = load(common.model_id(), adapter_path=args.adapter_path)
        m = classifier.evaluate(model, tok, rows)
        classifier.print_report(m, title=f"② ファインチューニング後 (adapter={args.adapter_path})")
        ft_acc = m["accuracy"]

    if base_acc is not None and ft_acc is not None:
        print("\n" + "=" * 56)
        print(f"accuracy: {base_acc*100:.1f}%  →  {ft_acc*100:.1f}%  "
              f"(差分 {(ft_acc-base_acc)*100:+.1f}pt)")
        print("=" * 56)


if __name__ == "__main__":
    main()
