#!/usr/bin/env bash
# WSI TITAN 実行ラッパー
# 使い方: ./run_titan.sh --input data/<cohort> [--dry-run] [--limit 2] [analyze_wsi_titan.py の引数]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
# user-space OpenSlide (sudo不要版)
export LD_LIBRARY_PATH="$HOME/.local/openslide/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

if [ ! -d "$ROOT/.venv-titan" ]; then
  echo "[run_titan] .venv-titan がありません。先に README の環境構築手順を実行してください。" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-titan/bin/activate"
exec python "$ROOT/analyze_wsi_titan.py" "$@"
