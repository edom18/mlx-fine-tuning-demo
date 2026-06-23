"""
Chapter 1 ハンズオン: 環境とモデルの動作確認。
- mlx / mlx_lm のバージョン
- 既定モデルをロードして1回だけ生成
うまくいけば「環境はOK」。エラーが出たら common.py の ALTERNATES を試す。

実行:
    uv run python scripts/00_check_env.py
"""
from __future__ import annotations
import sys, platform
import common


def main() -> None:
    print("=" * 56)
    print("環境チェック")
    print("=" * 56)
    print("Python      :", platform.python_version())
    print("Platform    :", platform.platform())

    try:
        import mlx.core as mx
        import mlx_lm
        print("mlx         :", mx.__version__ if hasattr(mx, "__version__") else "(loaded)")
        print("mlx_lm      :", mlx_lm.__version__)
    except Exception as e:  # noqa: BLE001
        print("\n[NG] mlx / mlx_lm の import に失敗:", e)
        print("→ `uv add mlx-lm` を実行してください。")
        sys.exit(1)

    mid = common.model_id()
    print("model       :", mid)
    print("\nモデルをロード中…（初回はダウンロードのため数分かかります）")

    from mlx_lm import load, generate
    try:
        model, tokenizer = load(mid)
    except Exception as e:  # noqa: BLE001
        print("\n[NG] モデルのロードに失敗:", repr(e))
        print("→ common.py の ALTERNATES を順に試してください:")
        for a in common.ALTERNATES:
            print("   COURSE_MODEL=", a, sep="")
        sys.exit(1)

    messages = [{"role": "user", "content": "こんにちは。あなたは何ができますか？一文で答えて。"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)
    print("\n--- モデルの応答 ---")
    print(out.strip())
    print("\n[OK] 環境は準備できました。次は scripts/01_next_token.py へ。")


if __name__ == "__main__":
    main()
