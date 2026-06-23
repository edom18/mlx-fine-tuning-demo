"""
分類と評価の共通ロジック（02_baseline_eval.py と 07_eval.py から使う）。
- classify_one: 1件を分類してモデルの生出力と正規化ラベルを返す
- evaluate    : データ全件を分類し、指標(accuracy, ラベル外出力率, 混同行列, label別P/R/F1)を計算
"""
from __future__ import annotations
from collections import defaultdict
import common


def classify_one(model, tokenizer, text: str, max_tokens: int = 16) -> tuple[str, str | None]:
    from mlx_lm import generate
    messages = common.build_messages(text)["messages"]  # system + user
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return raw.strip(), common.normalize_label(raw)


def evaluate(model, tokenizer, rows: list[dict], show_progress: bool = True) -> dict:
    labels = common.label_names()
    results = []
    for i, r in enumerate(rows, 1):
        raw, pred = classify_one(model, tokenizer, r["text"])
        results.append({"text": r["text"], "gold": r["label"], "raw": raw, "pred": pred})
        if show_progress:
            print(f"\r  分類中 {i}/{len(rows)}", end="", flush=True)
    if show_progress:
        print()

    n = len(results)
    correct = sum(1 for r in results if r["pred"] == r["gold"])
    out_of_label = sum(1 for r in results if r["pred"] is None)

    # 混同行列 conf[gold][pred]
    conf = defaultdict(lambda: defaultdict(int))
    for r in results:
        conf[r["gold"]][r["pred"] or "(ラベル外)"] += 1

    # label別 precision / recall / f1
    per_label = {}
    for lb in labels:
        tp = sum(1 for r in results if r["gold"] == lb and r["pred"] == lb)
        fp = sum(1 for r in results if r["gold"] != lb and r["pred"] == lb)
        fn = sum(1 for r in results if r["gold"] == lb and r["pred"] != lb)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lb] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "label_out_rate": out_of_label / n if n else 0.0,
        "confusion": {g: dict(d) for g, d in conf.items()},
        "per_label": per_label,
        "results": results,
    }


def print_report(metrics: dict, title: str = "", max_errors: int = 8) -> None:
    if title:
        print("\n" + "=" * 56)
        print(title)
        print("=" * 56)
    print(f"件数            : {metrics['n']}")
    print(f"accuracy        : {metrics['accuracy']*100:.1f}%")
    print(f"ラベル外出力率  : {metrics['label_out_rate']*100:.1f}%  (低いほど良い)")

    print("\nラベル別 precision / recall / f1 (support):")
    for lb, m in metrics["per_label"].items():
        print(f"  {lb:<12} P={m['precision']*100:5.1f}  R={m['recall']*100:5.1f}  "
              f"F1={m['f1']*100:5.1f}  (n={m['support']})")

    print("\n混同行列 (行=正解, 列=予測):")
    for gold, row in metrics["confusion"].items():
        cells = ", ".join(f"{p}:{c}" for p, c in sorted(row.items(), key=lambda x: -x[1]))
        print(f"  {gold:<12} → {cells}")

    errors = [r for r in metrics["results"] if r["pred"] != r["gold"]]
    if errors:
        print(f"\n誤判定の例（最大{max_errors}件）:")
        for r in errors[:max_errors]:
            print(f"  - 入力: {r['text']}")
            print(f"      正解={r['gold']} / 予測={r['pred']} / 生出力={r['raw']!r}")
