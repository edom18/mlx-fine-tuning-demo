"""
Chapter 11: seed.jsonl の全件を mlx_lm.server に順次投げ、LoRA適用後の
分類結果を一覧表示する（08_client.py のバッチ版 = 自動化）。

08_client.py は引数で1件のテキストを渡して確認するが、本スクリプトは
data/seed.jsonl（人手ラベル付き）をインプットにして全件を流し、
予測ラベルと正解ラベルを突き合わせて「LoRAの効き」を一目で確認する。

先に scripts/08_serve.sh を別ターミナルで起動しておくこと。

実行:
    uv run python scripts/08_client_batch.py
    uv run python scripts/08_client_batch.py --data data/seed.jsonl

注意:
    seed.jsonl は test 側に回る人手データ（評価リーク対策の test = seed）なので、
    ここで出る一致率は「サニティチェック」程度の位置づけ。厳密な前後比較は
    07_eval.py（混同行列・ラベル別P/R 込み）を使う。
"""
from __future__ import annotations
import argparse
import json
import urllib.error
import urllib.request
import common

URL = "http://localhost:8080/v1/chat/completions"
# 学習で作った LoRA アダプタの場所（config/lora.yaml の adapter_path と一致）。
ADAPTERS = str(common.ROOT / "adapters")


def classify(text: str) -> str:
    """08_client.py の classify と同一。1件を server に投げてラベル文字列を得る。"""
    payload = {
        "model": common.model_id(),
        "messages": [
            {"role": "system", "content": common.system_prompt()},
            {"role": "user", "content": text},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        # seed を渡して 0.31.3 の逐次(_serve_single)経路を踏ませる（08_client.py 参照）。
        "seed": 0,
        # 起動時の --adapter-path は server 0.31.3 が取りこぼすため、body 側で
        # アダプタを明示して LoRA を確実に効かせる（08_client.py の詳説コメント参照）。
        "adapters": ADAPTERS,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA / "seed.jsonl"),
                    help="インプットの jsonl（{text, label} 形式）")
    args = ap.parse_args()

    rows = common.read_jsonl(args.data)
    print(f"入力データ: {args.data}  ({len(rows)}件)")
    print(f"接続先   : {URL}\n")

    correct = n_gold = n_out = 0
    for i, r in enumerate(rows, 1):
        text = r["text"]
        gold = r.get("label")  # seed.jsonl には正解ラベルがある
        try:
            raw = classify(text)
        except urllib.error.URLError as e:
            raise SystemExit(
                f"\n[エラー] server に接続できません ({e}).\n"
                "        先に別ターミナルで `bash scripts/08_serve.sh` を起動してください。"
            )

        pred = common.normalize_label(raw)
        shown = pred if pred is not None else f"(ラベル外: {raw!r})"
        if pred is None:
            n_out += 1

        if gold is not None:
            n_gold += 1
            ok = pred == gold
            correct += ok
            print(f"[{i:>2}] {'✓' if ok else '✗'} 予測: {shown}  / 正解: {gold}")
        else:
            print(f"[{i:>2}]   予測: {shown}")
        print(f"      {text}")

    print("\n" + "=" * 56)
    if n_gold:
        print(f"一致      : {correct}/{n_gold}  ({correct / n_gold * 100:.1f}%)")
    print(f"ラベル外出力: {n_out}/{len(rows)}")
    print("=" * 56)


if __name__ == "__main__":
    main()
