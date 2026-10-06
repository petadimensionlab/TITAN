# lungcancer_wsi — TITAN による肺癌WSI順次解析

TITAN を使って肺癌の全スライド画像 (WSI, Hamamatsu `.ndpi`) を1枚ずつ順次解析し、
スライドレベル埋め込みの抽出・評価を行うスクリプトです。

- パイプライン: OpenSlide タイル抽出 → CONCH v1.5 パッチ特徴量 → TITAN スライド埋め込み
  → 線形プローブ / 交絡解析
- 検証環境: WSL2 (Ubuntu) + NVIDIA RTX 4090 / CUDA 12.1

> 患者データ (`data/`) と解析結果 (`results/`) は private のため `.gitignore` で除外しています。
> 本ディレクトリにはスクリプトとドキュメントのみを格納します。

## ファイル

| ファイル | 役割 |
|---|---|
| `analyze_wsi_titan.py` | メイン: WSIを順次解析しTITAN埋め込みを抽出 (レジューム・ラベルマップ対応) |
| `linear_probe.py` | 埋め込みの線形プローブ評価 (層化K-fold CV、2群/多群自動判定) |
| `report.py` | `summary.csv` + 埋め込みから PCA 散布図と HTML レポート生成 |
| `extract_metadata.py` | 症例ID / 検体種別 / スキャン日 / バッチを抽出 |
| `analyze_confounds.py` | 症例ID fold分離・バッチ交絡・同一患者検定 |
| `run_titan.sh` | 実行ラッパー (OpenSlideパス + venv) |
| `requirements-titan.txt` | 依存パッケージ |

## セットアップ

```bash
# 1) uv + Python 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv-titan
source .venv-titan/bin/activate
uv pip install -r requirements-titan.txt

# 2) OpenSlide (sudo不要: user-space に展開する例)
mkdir -p ~/.local/openslide && cd /tmp
apt download libopenslide0 openslide-tools libopenjp2-7 libtiff6 libjpeg-turbo8 libwebp7 liblerc4 libjbig0 libdeflate0
for d in *.deb; do dpkg-deb -x "$d" ~/.local/openslide/; done

# 3) TITAN 重み (gated: アクセス申請 + ログイン)
hf auth login   # MahmoodLab/TITAN への Access 承認が必要
```

`run_titan.sh` は `~/.local/openslide` のライブラリパスを自動設定します。

## 使い方

```bash
# 一覧確認 (torch不要)
python3 analyze_wsi_titan.py --input data/<cohort> --dry-run

# 埋め込み抽出
./run_titan.sh --input data/<cohort> --output results/<cohort>_TITAN

# ラベル map (ファイル名 prefix -> ラベル) を指定
./run_titan.sh --input data/<cohort> --label-map data/<cohort>/labels.json

# 線形プローブ
python linear_probe.py --emb-dir results/<cohort>_TITAN

# メタデータ抽出 / 交絡解析
python extract_metadata.py --input data/<cohort> --out results/<cohort>_TITAN/metadata.csv
python analyze_confounds.py --emb-dir results/<cohort>_TITAN

# レポート生成
python report.py --input results/<cohort>_TITAN/summary.csv --emb-dir results/<cohort>_TITAN
```

主なオプション: `--max-patches` (既定2000), `--batch-size`, `--device {auto,cuda,cpu}`,
`--limit N`, `--overwrite`, `--label-map <json>`, `--zeroshot` (プロンプトは `ZEROSHOT_PROMPTS` を編集)。

## 検証の要点

- 群間で高精度が得られても、検体種別やスキャン時期が交絡していれば病因シグナルとは言えない。
  本スクリプト群は、症例単位のfold分離 (`analyze_confounds.py`) と同一患者検定、
  検体量の統制によってそれを確認するために用いた。
- 匿名化のため、コホート名・症例数・群構成・スキャン日等は記載しない。

## 注意

- TITAN / CONCH の重みは [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) で gated。
- 研究用途 (非商用)。TITAN は CC-BY-NC-ND 4.0。
