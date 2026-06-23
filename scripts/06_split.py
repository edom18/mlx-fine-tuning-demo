"""
Chapter 7/8 ハンズオン: 学習用JSONLを組み立て、train/valid/test に分割する。

評価リーク対策(記事の核心):
  - test.jsonl は「人が手書きした seed」だけから作る（teacherの癖が混じらない真の評価セット）
  - train/valid は「合成データ(kept) + 残りのseed」から作る

出力(= mlx_lm.lora の --data に渡すディレクトリ): data/train.jsonl / valid.jsonl / test.jsonl
いずれも chat 形式 {"messages":[system, user, assistant(=正解ラベル)]}

実行:
    uv run python scripts/06_split.py
"""
from __future__ import annotations
import random
from collections import defaultdict
import common

TEST_PER_LABEL = 2     # 各ラベルから人手seedをこの数だけ test に回す
VALID_RATIO = 0.12     # 合成データのうち valid に回す割合


def by_label(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["label"]].append(r)
    return d


def to_chat(rows):
    return [common.build_messages(r["text"], r["label"]) for r in rows]


def main() -> None:
    rng = random.Random(0)
    labels = common.label_names()

    seed = common.read_jsonl(common.DATA / "seed.jsonl")
    kept_path = common.DATA / "kept.jsonl"
    kept = common.read_jsonl(kept_path) if kept_path.exists() else []
    if not kept:
        print("[警告] data/kept.jsonl が無い/空です。まず 04_synthesize→05_judge を実行してください。")
        print("       今回は seed だけで分割します（学習効果は限定的）。")

    seed_by = by_label(seed)
    kept_by = by_label(kept)

    test, train, valid = [], [], []
    for lb in labels:
        s = seed_by.get(lb, [])[:]
        rng.shuffle(s)
        test += s[:TEST_PER_LABEL]          # 人手seed → test
        train += s[TEST_PER_LABEL:]         # 残りのseed → train

        k = kept_by.get(lb, [])[:]
        rng.shuffle(k)
        n_val = max(1, int(len(k) * VALID_RATIO)) if k else 0
        valid += k[:n_val]
        train += k[n_val:]

    rng.shuffle(train); rng.shuffle(valid); rng.shuffle(test)

    common.write_jsonl(common.DATA / "train.jsonl", to_chat(train))
    common.write_jsonl(common.DATA / "valid.jsonl", to_chat(valid))
    common.write_jsonl(common.DATA / "test.jsonl", to_chat(test))

    print(f"train={len(train)}  valid={len(valid)}  test={len(test)}")
    print("test は人手seedのみ・train/validは合成中心 → 評価リークを避けた構成。")
    print("次は config/lora.yaml を使って学習: uv run mlx_lm.lora -c config/lora.yaml")


if __name__ == "__main__":
    main()
