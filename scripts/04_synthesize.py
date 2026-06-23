"""
Chapter 7 ハンズオン: 大きなモデル(teacher)で教師データを増やす。
seed.jsonl の手書き例を種に、ラベルごとに多様な問い合わせ文を量産する。

2つのバックエンド:
  --backend anthropic  : Claude API を teacher に使う（高品質・推奨）。要 ANTHROPIC_API_KEY
  --backend local      : 手元の大きめMLXモデルで自己生成（完全オフライン・品質はやや劣る）

出力: data/raw_synth.jsonl  ({"text","label","source":"synth"})

実行例:
    uv run python scripts/04_synthesize.py --backend anthropic --per-label 30
    COURSE_TEACHER=mlx-community/gemma-3n-E4B-it-lm-bf16 \
        uv run python scripts/04_synthesize.py --backend local --per-label 20
"""
from __future__ import annotations
import argparse, json, os, re
import common

INSTRUCTION = """あなたは日本語のカスタマーサポート問い合わせ文を作る作問者です。
カテゴリ「{label}」({definition}) に**明確に該当する**問い合わせ文を {n} 件作ってください。

条件:
- 実際のユーザーが書きそうな自然な日本語。敬語/タメ口/箇条書き/誤字まじり など文体を散らす。
- 長さもバラす（短い1文〜数文）。
- 他カテゴリと紛らわしい「境界ぎりぎりだが、このカテゴリが正解」の例も2〜3割混ぜる。
- 既存の例と丸かぶりしない。
- 出力は **JSON配列のみ**。各要素は文字列。前置き・説明・```は一切付けない。

既存の例(参考):
{examples}
"""

def _parse_array(txt: str) -> list[str]:
    txt = txt.strip()
    txt = re.sub(r"^```(json)?", "", txt).strip()
    txt = re.sub(r"```$", "", txt).strip()
    try:
        arr = json.loads(txt)
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        # 配列で返らなかった場合、行ごとに救済
        return [l.strip(" -・\t") for l in txt.splitlines() if len(l.strip()) > 4]


def gen_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def gen_local(prompt: str, _model_cache={}):
    from mlx_lm import load, generate
    mid = os.environ.get("COURSE_TEACHER", "mlx-community/gemma-3n-E4B-it-lm-bf16")
    if "m" not in _model_cache:
        print(f"  (teacher をロード: {mid})")
        _model_cache["m"], _model_cache["t"] = load(mid)
    model, tok = _model_cache["m"], _model_cache["t"]
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(model, tok, prompt=ids, max_tokens=1500, verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "local"], default="anthropic")
    ap.add_argument("--per-label", type=int, default=30)
    ap.add_argument("--teacher-model", default="claude-sonnet-4-6", help="anthropic backend のモデルID")
    args = ap.parse_args()

    spec = common.load_labels()
    seed = common.read_jsonl(common.DATA / "seed.jsonl")
    out_rows: list[dict] = []

    for lab in spec["labels"]:
        name, definition = lab["name"], lab["definition"]
        examples = "\n".join(f"- {r['text']}" for r in seed if r["label"] == name) or "- (なし)"
        prompt = INSTRUCTION.format(label=name, definition=definition, n=args.per_label, examples=examples)
        print(f"[{name}] {args.per_label}件を生成中…")
        raw = gen_anthropic(prompt, args.teacher_model) if args.backend == "anthropic" else gen_local(prompt)
        items = _parse_array(raw)
        for t in items:
            out_rows.append({"text": t, "label": name, "source": "synth"})
        print(f"  → {len(items)}件取得")

    common.write_jsonl(common.DATA / "raw_synth.jsonl", out_rows)
    print(f"\n合計 {len(out_rows)}件を data/raw_synth.jsonl に書き出しました。")
    print("次は scripts/05_judge.py で品質ゲートにかける。")


if __name__ == "__main__":
    main()
