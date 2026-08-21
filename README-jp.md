# TITAN

> **病理学のためのマルチモーダル ホールスライド基盤モデル** — *Nature Medicine*
>
> 元プロジェクト: [mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN)

[English README](./README.md)

---

## 概要

このリポジトリは [MahmoodLab/TITAN](https://github.com/mahmoodlab/TITAN) の
**フォーク** で、**Apple Silicon (MPS) GPU** で動作するよう設定し、
パッケージ管理を **uv** に移行したものです。元のアップストリーム README は
[README-original.md](./README-original.md) に保存しています。

**TITAN**（Transformer-based pathology **I**mage and **T**ext **A**lignment
**N**etwork）は、自己教師あり学習と視覚-言語アライメントを用いて、病理画像の
ROI やホールスライドを汎用的な特徴表現へ変換するマルチモーダル基盤モデルです。

- **論文**: [Nature Medicine (2025)](https://www.nature.com/articles/s41591-025-03982-3)
- **モデル重み**: [MahmoodLab/TITAN on Hugging Face](https://huggingface.co/MahmoodLab/TITAN)（gated — アクセス申請が必要）
- **ブログ**: [Building vision-language guided multimodal whole-slide TITAN](https://www.linkedin.com/pulse/building-vision-language-guided-multimodal-whole-slide-tong-ding-0hawe/)

---

## このフォークでの変更点

アップストリームとの主な差分は以下のとおりです。

### 1. uv による環境管理（venv）

アップストリームは `conda` + `pip install -e .` を使用。本フォークでは **uv**
を使用します。

- `pyproject.toml` に全ランタイム依存を記載
- `uv.lock` で解決済みの依存バージョンを固定
- `.venv/` は `uv` が作成・管理

### 2. Apple Silicon (MPS) GPU 対応

CUDA 専用のコードパスを全てデバイス非依存にし、Apple Silicon では PyTorch が
**MPS（Metal）バックエンド**で自動的に動作するようにしました。

- `titan/utils.py`:
  - `get_device()` — `cuda` > `mps` > `cpu` の順で利用可能なデバイスを返す
  - `amp_dtype(device)` — CUDA では `bfloat16`、MPS/CPU では `float16`
- `titan/finetune.py`:
  - `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler(device.type, ...)`
  - `torch.cuda.amp.autocast` → `torch.autocast(device_type=device.type, ...)`
- `notebooks/*.ipynb` と `README.md` も `get_device()` / `amp_dtype()` /
  デバイス非依存の `autocast` を使用するよう更新

### 3. モデル読み込みに必要な依存修正

- **`transformers>=4.46,<5` に固定** — TITAN のリモートコード
  （`modeling_titan.py`）は **transformers 5.x と非互換**
  （`AttributeError: 'Titan' object has no attribute 'all_tied_weights_keys'` で失敗）。
  `transformers 4.57.6` で動作確認済み（モデル読み込み: 159M params）。
- `[build-system]`（setuptools）を追加し、`uv sync` で `titan` パッケージを
  editable インストールできるようにした（`python -m titan.finetune` に必要）。
- ビルドを壊していた不正な空の `authors` メールアドレスを削除。

---

## インストール

### 前提条件

- **Apple Silicon**（M 系列チップ）の macOS — MPS GPU サポート用
- [uv](https://docs.astral.sh/uv/) のインストール（`brew install uv`）

### セットアップ

```bash
git clone <あなたのフォークURL>/TITAN.git
cd TITAN

# venv を作成し uv.lock から全依存をインストール
uv sync

# 環境を有効化
source .venv/bin/activate
```

> モデル重みは Hugging Face で **gated** です。
> https://huggingface.co/MahmoodLab/TITAN でアクセス申請後、初回使用前にログイン:
>
> ```bash
> huggingface-cli login
> ```

---

## デバイス / GPU の利用

デバイス選択は自動です。同じコードが **CUDA**・**MPS**（Apple Silicon GPU）・
**CPU** のいずれでも動作します:

```python
from titan.utils import get_device, amp_dtype

device = get_device()   # Apple Silicon では torch.device('mps')
print(device)           # mps
```

混合精度は `amp_dtype(device)`（CUDA では `bfloat16`、MPS/CPU では
`float16`）と `torch.autocast(device_type=device.type, ...)` により
デバイス非依存です。

---

## 推論の実行

```python
from transformers import AutoModel
from titan.utils import amp_dtype, get_device

device = get_device()
model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)

# ... パッチ特徴量と座標を読み込む（CLAM/CONCH での抽出結果など）...

with torch.autocast(device_type=device.type, dtype=amp_dtype(device)), torch.inference_mode():
    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
```

---

## finetune の実行

> **重要**: `python titan/finetune.py` ではなく `python -m titan.finetune` で
> 実行してください（前者は `titan` パッケージの import が解決できず失敗します）。

```bash
uv run python -m titan.finetune \
  --num_epochs 10 \
  --batch_size 1 \
  --num_workers 0 \
  --save_dir ./logs
```

同梱の `finetune.py` は **ダミーデータセット** で動作します。実際の学習には
`FinetuneDataset` クラスを独自のデータローダーに置き換えてください。

---

## 既知の注意点（このフォーク）

| 問題 | 対処 |
|---|---|
| `transformers` 5.x は TITAN リモートコードと非互換 | `transformers` は `<5` に固定。アップグレードしないこと |
| macOS では `DataLoader(num_workers>0)` がクラッシュ | `--num_workers 0` を使用 |
| `num_epochs<=2` だと最後に `best_model_weights` が `None` で TypeError | `--num_epochs >= 3`（デフォルトは 10）を使用 |

### MPS での実測性能（batch=1, fp16, forward+backward）

| スライドあたりパッチ数 | ms / step |
|---|---|
| 36 | ~13 |
| 100 | ~15 |
| 400 | ~33 |
| 900 | ~80 |

ほぼ線形（**パッチあたり約 0.08 ms**）。同梱ダミー finetune
（100 サンプル、10 エポック、早期停止）は MPS で **約 1 分** で完了します。

---

## デモノートブック

- `notebooks/inference_demo.ipynb` — スライド埋め込み抽出
- `notebooks/zeroshot_demo.ipynb` — ゼロショット分類
- `notebooks/linear_probe_demo.ipynb` — 線形プローブ評価

---

## ライセンスと利用条件

ⓒ Mahmood Lab。[CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.ja)
ライセンスで公開 — 非商用・学術研究目的のみ、適切な帰属表示付きで利用可。
TITAN モデルおよび派生物の商用利用・販売・収益化は禁止。モデルダウンロードには
Hugging Face への事前登録と利用条件への同意が必要です。完全な利用条件は
[README-original.md](./README-original.md) を参照してください。

## 参考文献

```
@article{ding2025multimodal,
  title={A multimodal whole-slide foundation model for pathology},
  author={Ding, Tong and Wagner, Sophia J and Song, Andrew H and Chen, Richard J and others},
  journal={Nature Medicine},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
