#!/usr/bin/env python3
"""共有設定 (cohorts.json) のローダー。

リポジトリには cohorts.example.json を同梱する。実際の設定 cohorts.json は
患者ディレクトリ名・症例名を含むため .gitignore 済み。手順:

    cp cohorts.example.json cohorts.json   # 自環境のパスに合わせて編集

設定ファイルの形式:
    root                省略時 = このファイルの親 (lungcancer_wsi)
    cohorts[]           {name, data_dir, results_dir, meta_csv?}
    tissue_dir          組織クラスタ解析の出力先 (省略可, root 相対)
    specimens            results_dir 名 -> 検体種別ラベル (レポート/アトラス用)

すべてのスクリプト (viewer / conch_zeroshot / tissue_analysis / make_atlas /
report_build) はこのモジュール経由でパスとcohorts名を得る。
"""
from __future__ import annotations

import json
from pathlib import Path

PKG = Path(__file__).resolve().parent


def load(path: Path | None = None) -> dict:
    p = path or PKG / "cohorts.json"
    if not p.exists():
        raise SystemExit(
            f"{p} が見つかりません。cohorts.example.json をコピーして作成してください。"
        )
    cfg = json.loads(p.read_text(encoding="utf-8"))
    root = Path(cfg.get("root", "."))
    if not root.is_absolute():
        root = (PKG / root).resolve()
    for c in cfg["cohorts"]:
        for k in ("data_dir", "results_dir"):
            if c.get(k):
                c[k] = str((root / c[k]).resolve())
        if c.get("meta_csv"):
            c["meta_csv"] = str((root / c["meta_csv"]).resolve())
    if cfg.get("tissue_dir"):
        cfg["tissue_dir"] = str((root / cfg["tissue_dir"]).resolve())
    cfg["_root"] = str(root)
    return cfg


def results_name(cfg: dict, results_dir: str) -> str:
    """results ディレクトリ名 -> cohorts 表示名 (specimens 参照用)。"""
    for c in cfg["cohorts"]:
        if Path(c["results_dir"]).name == results_dir:
            return c["name"]
    return results_dir
