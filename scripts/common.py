"""
共通設定とユーティリティ。
全スクリプトはここからモデルID・ラベル・system promptを読み込む。
"""
from __future__ import annotations
import json
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# .env の読み込み
#   全スクリプトが common を import するので、ここで一度だけ .env を読めば
#   04_synthesize / 05_judge の Anthropic() が ANTHROPIC_API_KEY を拾える。
#   探索順: course/.env → リポジトリ直下/.env。
#   既存の環境変数は上書きしない（シェルで export 済みならそちらを優先）。
#   python-dotenv 未導入でも import エラーで止めない（export 運用なら不要なため）。
# ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _env in (Path(__file__).resolve().parents[1] / ".env",
                 Path(__file__).resolve().parents[2] / ".env"):
        if _env.exists():
            load_dotenv(_env)
            break
except ModuleNotFoundError:
    pass

# ─────────────────────────────────────────────────────────────
# モデルID
#   この教材はテキスト分類だけを扱うので、Gemma 3n の「テキスト専用LM」
#   抽出版（リポジトリ名の "-lm-"）を使う。本体の any-to-any(画像/音声込み)
#   チェックポイントは mlx_lm.load() では読めない（重みが language_model.
#   配下に梱包され、KV共有層のパラメータがミスマッチする）ため必ず "-lm-" を選ぶ。
#   既定は 4bit 量子化版（省メモリ・高速・QLoRA向き）。読めない/精度を上げたい
#   場合は ALTERNATES に切り替える。環境変数 COURSE_MODEL で上書きできる。
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "mlx-community/gemma-3n-E2B-it-lm-4bit"
ALTERNATES = [
    "mlx-community/gemma-3n-E2B-it-lm-4bit",   # 既定・省メモリ（QLoRA向き / テキスト専用LM）
    "mlx-community/gemma-3n-E2B-it-lm-bf16",    # 同じE2Bのフル精度・学習品質重視（約10GB）
    "mlx-community/gemma-3n-E4B-it-lm-4bit",    # 一回り大きいE4B（精度重視・メモリに余裕があれば）
]

def model_id() -> str:
    return os.environ.get("COURSE_MODEL", DEFAULT_MODEL)

# ─────────────────────────────────────────────────────────────
# パス
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def load_labels() -> dict:
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))

def label_names() -> list[str]:
    return [l["name"] for l in load_labels()["labels"]]

# ─────────────────────────────────────────────────────────────
# system prompt（生成LLM方式の肝：ラベル文字列だけを出させる）
# ─────────────────────────────────────────────────────────────
def system_prompt() -> str:
    spec = load_labels()
    lines = [f'- {l["name"]}: {l["definition"]}' for l in spec["labels"]]
    names = " / ".join(label_names())
    return (
        "あなたは社内SaaSのカスタマーサポート問い合わせ分類器です。\n"
        "ユーザーの問い合わせ文を読み、次のラベルのうち最も適切なものを1つだけ、"
        "ラベル名のみで出力してください。説明・前置き・記号・引用符は一切付けないこと。\n\n"
        "ラベル定義:\n" + "\n".join(lines) + "\n\n"
        f"出力できるのは次のいずれか1つだけです: {names}"
    )

def build_messages(text: str, label: str | None = None) -> dict:
    """1件分の chat 形式（学習用は label を渡す / 推論用は None）。"""
    msgs = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": text},
    ]
    if label is not None:
        msgs.append({"role": "assistant", "content": label})
    return {"messages": msgs}

def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_label(raw: str) -> str | None:
    """モデル出力をラベル集合へ寄せる。完全一致しなければ None（=ラベル外出力）。"""
    s = (raw or "").strip().strip("「」\"' 　。.")
    names = label_names()
    if s in names:
        return s
    # 部分一致の救済（"請求" → "請求・支払い" 等）。厳密評価では使わない。
    for n in names:
        if n in s or s in n:
            return n
    return None
