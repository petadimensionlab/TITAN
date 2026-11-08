#!/usr/bin/env bash
# TITAN WSI 統合ビューワ 起動スクリプト
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/.local/openslide/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$(dirname "$0")/../.venv-viewer/bin/python" app.py
