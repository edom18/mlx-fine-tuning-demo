# mlx-fine-tuning-demo

Apple Silicon Mac だけで完結する、ローカル LLM ファインチューニングのデモです。
**Gemma 3n E2B** を **MLX** で LoRA ファインチューニングし、ローカル運用（API サーバ / Ollama / llama.cpp）まで一周します。

題材は **カスタマーサポート問い合わせの 6 分類**。分類ヘッドは付けず、モデルに **ラベル文字列そのものを生成させる「生成 LLM 方式」** で解きます。

> このリポジトリは解説ブログ記事の補足です。記事を読みながら、実際に手元で同じパイプラインを動かせるようにまとめてあります。
> 記事URL: <!-- ここにブログ記事のURLを記載 -->

---

## これは何をするデモか

`scripts/` に番号順のパイプラインが入っています。素のモデルの実力を測り（ベースライン）、教師データを合成・品質審査し、LoRA で学習し、学習前後を同じ物差しで比較し、最後にローカル運用する——という一連の流れを再現できます。

```
seed.jsonl(手書き)
   │  ① ベースライン測定
   │  ② teacher で教師データ合成 → judge で品質ゲート
   │  ③ train/valid/test に分割（評価リーク対策つき）
   ▼
LoRA 学習（adapters/）
   │  ④ 学習前後を比較評価
   ▼
ローカル運用（mlx_lm.server / Ollama / llama.cpp）
```

---

## 前提

- **Apple Silicon Mac**（M1 以降）。検証環境は M4 / 32GB。
- 学習は **MLX（`mlx-lm`）** を使います。
  Unsloth / bitsandbytes QLoRA は **NVIDIA 専用**で Mac では動きません。
- [**uv**](https://docs.astral.sh/uv/) が入っていること（Python とパッケージはこれで揃います）。
- 教師データ合成・審査で **Claude API** を使う場合のみ `ANTHROPIC_API_KEY` が必要です（完全オフラインでも実行可能。後述）。

> モデルは必ず**テキスト専用 LM 版**（リポジトリ名に `-lm-` が入るもの。既定は `mlx-community/gemma-3n-E2B-it-lm-4bit`）を使います。Gemma 3n の本体（画像・音声込みの any-to-any 版）は重みが `language_model.` 配下に梱包されており `mlx_lm.load()` では読めません。

---

## セットアップ

```bash
git clone <このリポジトリのURL>
cd mlx-fine-tuning-demo

uv sync                  # 依存をインストール（mlx-lm など）。初回は少し時間がかかります

# Claude API を teacher に使う場合だけ（オフライン実行なら不要）
cp .env.example .env     # .env を開いて ANTHROPIC_API_KEY を設定
```

---

## 動かす（ステップ順）

各スクリプトはプロジェクトルートから `uv run` で実行します。

| # | コマンド | 何をするか |
|---|----------|-----------|
| 1 | `uv run python scripts/00_check_env.py` | 環境とモデルのロード確認（初回はモデルDLで数分） |
| 2 | `uv run python scripts/01_next_token.py` | 「LLM は次トークン予測器」を可視化 |
| 3 | `uv run python scripts/02_baseline_eval.py` | 素のモデルの分類力を測る（**ベースライン**。この数字を覚えておく） |
| 4 | `uv run python scripts/04_synthesize.py --backend anthropic --per-label 30` | teacher で教師データを増幅 → `data/raw_synth.jsonl` |
| 5 | `uv run python scripts/05_judge.py --backend anthropic` | judge で品質ゲート → `data/kept.jsonl` / `rejected.jsonl` |
| 6 | `uv run python scripts/06_split.py` | train/valid/test に分割（リーク対策つき） |
| 7 | `uv run mlx_lm.lora -c config/lora.yaml` | **LoRA 学習** → `adapters/` |
| 8 | `uv run python scripts/07_eval.py` | 学習前後を同じ物差しで比較 |
| 9 | `bash scripts/08_serve.sh` ＋ `uv run python scripts/08_client.py "領収書を再発行したい"` | mlx_lm.server で運用（**推奨**） |

> **完全オフラインで進めたい場合**は、ステップ 4・5 で `--backend local` を指定します（teacher / judge に手元の大きめ MLX モデルを使う。品質はやや劣りますが API キー不要）。
> teacher モデルは環境変数 `COURSE_TEACHER` / `COURSE_JUDGE` で変更できます。

### 運用のバリエーション

- **mlx_lm.server（推奨・最も確実）**: `bash scripts/08_serve.sh` で OpenAI 互換 API が `http://localhost:8080` に立ちます。別ターミナルから `scripts/08_client.py`（1件）や `scripts/08_client_batch.py`（seed 全件）で叩けます。
- **Ollama**: `bash scripts/09_to_ollama.sh`（アダプタ融合 → GGUF 変換 → `Modelfile` で登録）。対応アーキのみ。
- **llama.cpp / LLM for Unity**: `bash scripts/10_to_gguf.sh`（融合 → `convert_hf_to_gguf` → 量子化。1ファイルの量子化 GGUF を出力）。

---

## ディレクトリ構成

```
.
├── scripts/            番号順のパイプライン（共有ロジックは番号なしの common.py / classifier.py）
├── config/lora.yaml    LoRA 学習設定（mlx_lm.lora -c で渡す）
├── data/
│   ├── labels.json     6ラベルの定義・境界ケース
│   └── seed.jsonl      人手で書いた種データ（評価 test はここからのみ作る）
├── Modelfile           Ollama 登録用
├── pyproject.toml      依存定義（uv）
└── uv.lock             ロックファイル（再現性のためコミット済み）
```

### 生成物（`.gitignore` 済み・再生成可能）

- `data/raw_synth.jsonl` … 合成直後（未審査）
- `data/kept.jsonl` / `rejected.jsonl` … judge 後
- `data/train.jsonl` / `valid.jsonl` / `test.jsonl` … 学習・評価用（chat 形式）
- `adapters/` … 学習した LoRA アダプタ（本体モデルは書き換えない）
- `fused_model/` … アダプタ融合済みモデル（Ollama / llama.cpp 用）

---

## 設計のポイント（記事の核心）

- **生成 LLM 方式**: 出力層は語彙数のまま、ラベル文字列を生成させる。分類ヘッドは付けない。
- **損失は assistant 部分だけに**: `config/lora.yaml` の `mask_prompt: true`。正解ラベル部分だけを学習対象にする SFT の肝。
- **評価リーク対策**: `test.jsonl` は人手 `seed.jsonl` のみから作り、`train/valid` は合成データ中心にする。teacher の癖が評価に混じらないようにする。
- **評価は accuracy 単独で終わらせない**: 混同行列 / ラベル別 P・R・F1 / ラベル外出力率 / 誤判定例まで見る（`scripts/07_eval.py`）。
- **学習・推論・運用で同じ system prompt / ラベル定義を使う**: `scripts/common.py` と `data/labels.json` に集約。条件をズラさない。

---

## 困ったら

- **モデルのロードで落ちる** → `scripts/common.py` の `ALTERNATES` を順に試す
  （例: `COURSE_MODEL=mlx-community/gemma-3n-E2B-it-lm-bf16 uv run python scripts/00_check_env.py`）。
  必ずテキスト専用 LM 版（`-lm-`）を使うこと。
- **メモリ不足** → `config/lora.yaml` の `batch_size` / `max_seq_length` / `num_layers` を下げる。
- **GGUF 変換に失敗** → `scripts/08_serve.sh`（mlx_lm.server）で運用する。機能はローカル運用として同等。
- **サーバ推論が無応答／落ちる（gemma3n）** → クライアントのリクエストに `"seed"` を入れる（`08_client.py` は対応済み）。
  mlx_lm.server が gemma3n を batched 生成と誤判定し、KV 共有層でアンパックに失敗するため、`seed` で逐次生成にフォールバックさせる。

---

## ライセンス / 注意

学習・実行には外部モデル（Gemma 3n）と、任意で Claude API を使います。各モデル・API の利用規約に従ってください。
