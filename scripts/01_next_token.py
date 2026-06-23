"""
Chapter 2 ハンズオン: 「LLMは次トークン予測器」を自分の目で見る。
あるプロンプトに対して、次に来るトークンの確率トップNを表示する。
分類でもなんでもなく、語彙の中から次の1トークンを選んでいるだけ、という感覚をつかむ。

実行:
    uv run python scripts/01_next_token.py
    uv run python scripts/01_next_token.py "日本の首都は"
"""
from __future__ import annotations
import sys
import numpy as np
import mlx.core as mx
from mlx_lm import load
import common


def top_next_tokens(model, tokenizer, prompt_ids, k: int = 15):
    logits = model(mx.array(prompt_ids)[None])      # [1, seq_len, vocab]
    last = logits[0, -1, :]                          # 最後の位置の語彙ロジット
    probs = mx.softmax(last.astype(mx.float32), axis=-1)
    probs = np.array(probs)
    idx = probs.argsort()[::-1][:k]                  # 上位k
    return [(int(i), tokenizer.decode([int(i)]), float(probs[i])) for i in idx]


def main() -> None:
    user_text = sys.argv[1] if len(sys.argv) > 1 else "問い合わせ「領収書を再発行したい」の分類は"
    model, tokenizer = load(common.model_id())

    # chat テンプレートを通さず、素のテキストの続きを見る（次トークン予測の生の姿）
    ids = tokenizer.encode(user_text)
    print(f"\nプロンプト: {user_text!r}")
    print(f"トークン数: {len(ids)}")
    print("\n次トークン候補（確率トップ15）:")
    print("-" * 44)
    for rank, (tid, piece, p) in enumerate(top_next_tokens(model, tokenizer, ids), 1):
        bar = "█" * max(1, int(p * 40))
        shown = piece.replace("\n", "\\n")
        print(f"{rank:>2}. {p*100:6.2f}%  {bar:<40} {shown!r}")

    print("\n要点: モデルは『分類』しているのではなく、語彙(数万トークン)の上に")
    print("      確率分布を作って、最も尤もらしい次の1トークンを選んでいるだけ。")
    print("      生成LLM方式のSFTは、この確率の山を『正解ラベル文字列』へ寄せる作業。")


if __name__ == "__main__":
    main()
