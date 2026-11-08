#!/usr/bin/env python3
"""クラス別代表パッチのアトラス画像生成 (C-2 画像アトラス版用).

conch_zeroshot.py の分類処理を再利用し, 対象コホートの全グリッドを再推論して,
7クラスそれぞれについて確率が高い代表パッチ (別スライド優先で最大2例) を選ぶ.
コホートは ATLAS_COHORT 環境変数 (results ディレクトリ名) で選択 (デフォルト: 最初の cohorts).

出力: viewer/atlas/
  patch_<CLS>_<n>_<stem>.png     ... 512px パッチ + 100µm スケールバー
  overview_<stem>.png            ... 低倍率全景 + 選択パッチに赤枠
  atlas_manifest.json            ... 選択一覧 (スライド・座標・確率・ファイル)

実行: LD_LIBRARY_PATH=$HOME/.local/openslide/usr/lib/aarch64-linux-gnu \
      python viewer/make_atlas.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import openslide
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conch_zeroshot as cz  # noqa: E402

ROOT = cz.ROOT
RES = cz.RES
OUT = Path(__file__).resolve().parent / "atlas"
# 対象コホート (results ディレクトリ名)。cz.COHORT_DATA のキーから環境変数で選択可能。
COHORT = os.environ.get("ATLAS_COHORT") or next(iter(cz.COHORT_DATA))
TOP_PER_SLIDE = 3                 # スライドごと1クラスに保持する候補数
FINAL_PER_CLASS = 2               # 最終的に選ぶ例の数
CONTEXT = 512                     # 全景図での枠の半幅 (level0 px)


def main() -> None:
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cz.build_model(cz.DEFAULT_WEIGHTS).to(dev).eval()
    W = cz.build_text_weights(model, dev)
    scale = model.logit_scale.exp().to(dev).detach()
    transform = cz.make_transform()

    _mc = next((c["meta_csv"] for c in cz.CFG["cohorts"] if c.get("meta_csv")), None)
    meta_csv = Path(_mc) if _mc else RES / "metadata.csv"
    meta = {}
    if meta_csv.exists():
        meta = {r["stem"].replace(" ", "_"): r
                for r in csv.DictReader(meta_csv.open(encoding="utf-8"))}

    res = RES / COHORT
    data_dir = Path(cz.COHORT_DATA[COHORT])
    ndpi_idx = {cz.stem_key(p.name): p for p in data_dir.glob("*.ndpi")
                if not p.name.startswith("._")}

    # クラスごとに (確率, stem, x, y) の候補プール
    pool: dict[str, list[tuple[float, str, int, int]]] = defaultdict(list)
    for h5p in sorted(res.glob("*.h5")):
        st = h5p.stem
        ndpi = ndpi_idx.get(st)
        if ndpi is None:
            print(f"SKIP {st}: NDPI 無し")
            continue
        with h5py.File(h5p, "r") as f:
            coords = f["coords"][:].astype(np.int64)
        lab, prob, _ = cz.classify_slide(ndpi, coords, model, transform, W, scale, dev)
        for i, (cid, _, _) in enumerate(cz.CLASSES):
            sel = np.argsort(prob[lab == i])[-TOP_PER_SLIDE:]
            sub = np.flatnonzero(lab == i)[sel]
            for j in sub:
                pool[cid].append((float(prob[j]), st, int(coords[j][0]), int(coords[j][1])))
        print(f"  {st}: n={len(lab)} done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    chosen_by_slide: dict[str, list[tuple[int, int]]] = defaultdict(list)
    ids = {c[0]: i for i, c in enumerate(cz.CLASSES)}
    for cid, _, _ in cz.CLASSES:
        cand = sorted(pool[cid], key=lambda t: -t[0])
        picked, seen_stems = [], set()
        for p in cand:                       # 別スライド優先
            if p[1] not in seen_stems:
                picked.append(p)
                seen_stems.add(p[1])
            if len(picked) == FINAL_PER_CLASS:
                break
        if len(picked) < FINAL_PER_CLASS:    # 足りなければ同一スライド許容で補充
            for p in cand:
                if p not in picked:
                    picked.append(p)
                if len(picked) == FINAL_PER_CLASS:
                    break
        for n, (pr, st, x, y) in enumerate(picked, 1):
            chosen_by_slide[st].append((x, y))
            mpp = float(meta.get(st, {}).get("mpp_x") or 0.4545)
            fname = f"patch_{cid}_{n}_{st}.png"
            crop_patch(ndpi_idx[st], x, y, mpp, OUT / fname)
            manifest.append({"class": cid, "n": n, "stem": st, "x": x, "y": y,
                             "prob": round(pr, 3), "mpp": mpp, "patch": fname})
            print(f"{cid}#{n}: {st} ({x},{y}) prob={pr:.3f}")

    # 全景図 (選択パッチに赤枠)
    for st, boxes in chosen_by_slide.items():
        overview(ndpi_idx[st], boxes, OUT / f"overview_{st}.png")

    (OUT / "atlas_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}: {len(manifest)} patches, {len(chosen_by_slide)} overviews")


def crop_patch(ndpi: Path, x: int, y: int, mpp: float, out: Path) -> None:
    """512px パッチ (+100µm スケールバー) を保存."""
    slide = openslide.OpenSlide(str(ndpi))
    try:
        mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER,
                                         20) or 20)
    except (ValueError, TypeError):
        mag = 20.0
    read = int(round(512 * (mag / 20.0))) if mag else 512
    w, h = slide.dimensions
    x0 = max(0, min(x, w - read))
    y0 = max(0, min(y, h - read))
    img = slide.read_region((x0, y0), 0, (read, read)).convert("RGB")
    slide.close()
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BILINEAR)
    bar = ImageDraw.Draw(img)
    # 100µm を level0 px に換算し、512px へ縮小後の表示pxへ揃える
    bar_len = int(round(100.0 / mpp * 512 / read))
    bar_len = max(20, min(bar_len, 400))    # 100µm を 512px 系に換算
    bar.rectangle([20, 484 - 8, 20 + bar_len, 484], fill="black",
                  outline="white", width=1)
    img.save(out)


def overview(ndpi: Path, boxes: list[tuple[int, int]], out: Path) -> None:
    """低倍率の全景に選択パッチ位置へ赤枠を描く."""
    slide = openslide.OpenSlide(str(ndpi))
    lvl = slide.get_best_level_for_downsample(8.0)
    ds = slide.level_downsamples[lvl]
    img = slide.read_region((0, 0), lvl, slide.level_dimensions[lvl]).convert("RGB")
    slide.close()
    d = ImageDraw.Draw(img)
    side = int(round(2 * CONTEXT / ds))
    for x, y in boxes:
        cx, cy = (x + 256) / ds, (y + 256) / ds
        w = max(4, img.width // 200)
        d.rectangle([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2],
                    outline=(200, 16, 46), width=w)
    if img.width > 1400:
        img = img.resize((1400, int(img.height * 1400 / img.width)), Image.BILINEAR)
    img.save(out)


if __name__ == "__main__":
    main()
