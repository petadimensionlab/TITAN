# lungcancer_wsi — TITAN による肺癌WSI順次解析

TITAN を使って肺癌の全スライド画像 (WSI, Hamamatsu `.ndpi`) を1枚ずつ順次解析し、
スライドレベル埋め込みの抽出・評価を行うスクリプトです。

- 対象: 肺癌切除/生検 WSI (DrKitai, DrHatanaka コホート)
- パイプライン: OpenSlide タイル抽出 → CONCH v1.5 パッチ特徴量 → TITAN スライド埋め込み → 線形プローブ / 交絡解析
- 検証環境: WSL2 (Ubuntu) + NVIDIA RTX 4090 / CUDA 12.1

> 患者データ (`data/`) と解析結果 (`results/`) は private のため `.gitignore` で除外しています。
> 本ディレクトリにはスクリプトとドキュメントのみを格納します。

## ファイル

| ファイル | 役割 |
|---|---|
| `analyze_drkitai_titan.py` | メイン: WSIを順次解析しTITAN埋め込みを抽出 (レジューム・ラベルマップ対応) |
| `linear_probe_drkitai.py` | 埋め込みの線形プローブ評価 (層化K-fold CV、2群/多群自動判定) |
| `report_drkitai.py` | `summary.csv` + 埋め込みから PCA 散布図と HTML レポート生成 |
| `extract_metadata_drhatanaka.py` | 症例ID / 検体種別 / スキャン日 / バッチを抽出 |
| `analyze_confounds_drhatanaka.py` | 症例ID fold分離・バッチ交絡・同一患者検定 |
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
python3 analyze_drkitai_titan.py --dry-run

# DrKitai (KRAS G12C/other/wild) — 埋め込み + ゼロショット
./run_titan.sh --input data/DrKitai --output results/DrKitai_TITAN --zeroshot

# DrHatanaka (術後再発 / 4期CRT後再発) — ラベルマップ指定
./run_titan.sh --input data/DrHatanaka \
  --label-map data/DrHatanaka/labels.json \
  --output results/DrHatanaka_TITAN

# 線形プローブ
python linear_probe_drkitai.py --emb-dir results/DrHatanaka_TITAN

# レポート生成
python report_drkitai.py --input results/DrHatanaka_TITAN/summary.csv \
  --emb-dir results/DrHatanaka_TITAN
```

主なオプション: `--max-patches` (既定2000), `--batch-size`, `--device {auto,cuda,cpu}`,
`--limit N`, `--overwrite`, `--label-map <json>`, `--zeroshot`。

## 実行結果

### DrKitai (58枚, KRAS 3群: G12C 22 / other 15 / wild 21)

- 抽出 `ok=58 fail=0`、埋め込み768次元、所要 **10分20秒** (RTX 4090)
- ゼロショット: acc 0.207 / 線形プローブ: acc 0.329, AUROC 0.601 (実質チャンスレベル)

### DrHatanaka (45枚, 術後再発26 / 4期CRT後再発19)

- 抽出 `ok=45 fail=0`、所要 **13分9秒**
- 線形プローブ: acc 0.978 / AUROC 0.980
- 全例300patchに統一 (36例): acc 0.946 / AUROC 1.000
- 症例ID fold分離 (StratifiedGroupKFold): acc 0.958 / AUROC 0.980
- 同一患者検定: 最近傍一致 0.044 (chance 0.040) → 患者同一性の漏洩なし

**重要:** DrHatanaka は検体種別 (切除 vs 生検) とスキャン期間 (2025-10〜12 vs 2026-2〜4) が
完全に交絡しています。高精度を再発生物学へ帰属することはできません。

## 注意

- TITAN / CONCH の重みは [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) で gated。
- 研究用途 (非商用)。TITAN は CC-BY-NC-ND 4.0。
