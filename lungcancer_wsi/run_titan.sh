#!/usr/bin/env bash
# DrKitai TITAN 高級実行ラッパー
# 使い方: ./run_titan.sh [--dry-run] [--limit 2] [その他 analyze_drkitai_titan.py 引数]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
# user-space OpenSlide (sudo不要版)
export LD_LIBRARY_PATH="$HOME/.local/openslide/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# Strata API はWSL内127.0.0.1不可のためgateway経由 (TITAN実行自体には不要、LLM連携用)
export STRATA_GATEWAY="$(ip route show | grep -oP '(?<=via )[^ ]+' | head -n1)"
export OPENAI_BASE_URL="http://${STRATA_GATEWAY:-192.168.208.1}:8080/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"

if [ ! -d "$ROOT/.venv-titan" ]; then
  echo "[run_titan] .venv-titan がありません。先に README の環境構築手順を実行してください。" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-titan/bin/activate"
exec python "$ROOT/analyze_drkitai_titan.py" "$@"
