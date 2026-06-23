"""
Chapter 11: 立てた mlx_lm.server をアプリから叩く例(OpenAI互換)。
標準ライブラリだけで動く。先に scripts/08_serve.sh を別ターミナルで起動しておく。

実行:
    uv run python scripts/08_client.py "領収書を再発行したいです"
"""
from __future__ import annotations
import json, sys, urllib.request
import common

URL = "http://localhost:8080/v1/chat/completions"
# 学習で作った LoRA アダプタの場所（config/lora.yaml の adapter_path と一致）。
# server を別マシンで立てる場合はそのマシン上のパスに読み替えること。
ADAPTERS = str(common.ROOT / "adapters")


def classify(text: str) -> str:
    payload = {
        "model": common.model_id(),
        "messages": [
            {"role": "system", "content": common.system_prompt()},
            {"role": "user", "content": text},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        # seed を渡すと mlx_lm.server(0.31.3) は「逐次(_serve_single)」経路で応答する。
        # これが無いと server は batched 生成を選び、gemma3n の KV共有層が
        # `keys, values = cache.state` でアンパックに失敗して落ちる(worker例外で無応答)。
        # temperature=0.0 の決定論分類なので固定 seed は無害。
        "seed": 0,
        # mlx_lm.server 0.31.3 は起動時の --adapter-path を ModelProvider.load() 内で
        # 取りこぼし(model名を解決後のキーで _adapter_map を引くため必ず外れる)、素の
        # ベースモデルを配信してしまう。リクエスト body の "adapters" は load() の
        # デフォルト引数側に渡るため確実に効く。これが無いと LoRA 未適用の結果になる。
        "adapters": ADAPTERS,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "ログインできません。パスワードを再設定したいです。"
    label = classify(text)
    print(f"入力: {text}")
    print(f"分類: {label}  (正規化: {common.normalize_label(label)})")
