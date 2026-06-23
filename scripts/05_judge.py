"""
Chapter 7 ハンズオン: judge(審判) で不良データを弾く品質ゲート。
合成した raw_synth.jsonl の各件について「この文は本当にこのラベルが正解か？」を
別モデルに二択で判定させ、合格だけを残す。

ポイント(記事の核心): データは多ければよいのではない。低品質データはモデルを壊す。
  - ラベルが曖昧/誤り → 捨てる
  - 重複 → 捨てる

2バックエンド（synthesizeと同じ）:
  --backend anthropic / --backend local

出力:
  data/kept.jsonl      合格データ
  data/rejected.jsonl  不合格データ(理由つき。目視レビュー用)

実行:
    uv run python scripts/05_judge.py --backend anthropic
"""
from __future__ import annotations
import argparse, json, re
import common

JUDGE_PROMPT = """次の問い合わせ文に対して、付与されたラベルが正しいか判定してください。

ラベル定義:
{defs}

判定対象:
  問い合わせ: {text}
  付与ラベル: {label}

次のJSONだけを出力（説明禁止）:
{{"verdict": "ok" または "ng", "correct_label": "最も正しいと思うラベル名", "reason": "短い理由"}}
"""

def _parse_json(txt: str) -> dict | None:
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def judge_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    msg = client.messages.create(model=model, max_tokens=300,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def judge_local(prompt: str, _c={}):
    from mlx_lm import load, generate
    import os
    mid = os.environ.get("COURSE_JUDGE", "mlx-community/gemma-3n-E4B-it-lm-bf16")
    if "m" not in _c:
        print(f"  (judge をロード: {mid})")
        _c["m"], _c["t"] = load(mid)
    ids = _c["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_c["m"], _c["t"], prompt=ids, max_tokens=200, verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "local"], default="anthropic")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    spec = common.load_labels()
    defs = "\n".join(f'- {l["name"]}: {l["definition"]}' for l in spec["labels"])
    rows = common.read_jsonl(common.DATA / "raw_synth.jsonl")

    kept, rejected, seen = [], [], set()
    for i, r in enumerate(rows, 1):
        # 重複除去
        key = r["text"].strip()
        if key in seen:
            rejected.append({**r, "reason": "重複"}); continue
        seen.add(key)

        prompt = JUDGE_PROMPT.format(defs=defs, text=r["text"], label=r["label"])
        raw = judge_anthropic(prompt, args.judge_model) if args.backend == "anthropic" else judge_local(prompt)
        v = _parse_json(raw) or {"verdict": "ng", "reason": "judge出力をパースできず"}
        if v.get("verdict") == "ok" and v.get("correct_label", r["label"]) == r["label"]:
            kept.append(r)
        else:
            rejected.append({**r, "reason": v.get("reason", ""), "suggested": v.get("correct_label")})
        print(f"\r  審査中 {i}/{len(rows)}  合格={len(kept)} 不合格={len(rejected)}", end="", flush=True)
    print()

    common.write_jsonl(common.DATA / "kept.jsonl", kept)
    common.write_jsonl(common.DATA / "rejected.jsonl", rejected)
    print(f"合格 {len(kept)} / 不合格 {len(rejected)}  → data/kept.jsonl, data/rejected.jsonl")
    print("rejected.jsonl を一度は目で見ること（judgeも完璧ではない）。次は 06_split.py。")


if __name__ == "__main__":
    main()
