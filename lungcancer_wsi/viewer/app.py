#!/usr/bin/env python3
"""TITAN 解析結果 × WSI 画像 統合ビューワ.

data/ の NDPI 画像と results/ の TITAN 出力 (summary.csv, .h5, .pt) を
ファイル名で突合し, スライド表示 + パッチ座標オーバーレイ + メタデータ表示を行う.

起動: ./run_viewer.sh  (http://localhost:8700)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import openslide
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from PIL import Image

PKG = Path(__file__).resolve().parent.parent      # lungcancer_wsi/
STATIC = Path(__file__).resolve().parent / "static"
TILE = 256

import sys
sys.path.insert(0, str(PKG))
import cohorts as cohorts_cfg  # noqa: E402

# cohorts.json (gitignore 済み) からコホート定義を読む。cohorts.example.json 参照。
# root = data/ results/ を含む作業ツリーの最上位ディレクトリ。
CFG = cohorts_cfg.load()
ROOT = Path(CFG["_root"])
CACHE = ROOT / ".viewer_cache"
COHORTS = CFG["cohorts"]

app = Flask(__name__, static_folder=None)


def stem_key(name: str) -> str:
    """'Example Slide - 2025-12-01 15.49.12.ndpi' -> 'Example_Slide_-_2025-12-01_15.49.12'"""
    return Path(name).stem.replace(" ", "_")


# ---------------------------------------------------------------- index ---
def build_index() -> list[dict]:
    slides: list[dict] = []
    sid = 0
    for cohort in COHORTS:
        data_dir = ROOT / cohort["data_dir"]
        res_dir = ROOT / cohort["results_dir"]

        summary: dict[str, dict] = {}
        scsv = res_dir / "summary.csv"
        if scsv.exists():
            with scsv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    summary[stem_key(row.get("filename", ""))] = row

        seen: set[str] = set()
        imgs = sorted(
            p for p in data_dir.glob("*")
            if p.suffix.lower() in {".ndpi", ".svs", ".tif", ".tiff"}
            and not p.name.startswith("._")
        ) if data_dir.exists() else []
        for p in imgs:
            key = stem_key(p.name)
            seen.add(key)
            h5 = res_dir / f"{key}.h5"
            pt = res_dir / f"{key}.pt"
            slides.append(
                _record(
                    sid, cohort, str(p.relative_to(ROOT)), key,
                    h5 if h5.exists() else None,
                    pt if pt.exists() else None,
                    summary.get(key),
                )
            )
            sid += 1

        # 結果のみ (画像が見つからない) レコード
        for key, row in sorted(summary.items()):
            if key in seen:
                continue
            h5 = res_dir / f"{key}.h5"
            pt = res_dir / f"{key}.pt"
            if not (h5.exists() or pt.exists()):
                continue
            slides.append(
                _record(
                    sid, cohort, row.get("relative_path", key), key,
                    h5 if h5.exists() else None,
                    pt if pt.exists() else None,
                    row,
                )
            )
            sid += 1
    return slides


def _record(
    sid: int, cohort: dict, rel_path: str, key: str,
    h5: Path | None, pt: Path | None, row: dict | None,
) -> dict:
    rec = {
        "id": sid,
        "cohort": cohort["name"],
        "name": Path(rel_path).name,
        "key": key,
        "path": rel_path,
        "has_image": (ROOT / rel_path).exists(),
        "has_h5": h5 is not None,
        "has_pt": pt is not None,
        "has_report": (ROOT / cohort["results_dir"] / "report.html").exists(),
        "h5": str(h5.relative_to(ROOT)) if h5 else None,
        "pt": str(pt.relative_to(ROOT)) if pt else None,
        "label": (row or {}).get("label", ""),
        "status": (row or {}).get("status", ""),
        "n_patches": (row or {}).get("n_patches", ""),
        "size_mb": (row or {}).get("size_mb", ""),
        "seconds": (row or {}).get("seconds", ""),
        "pred": (row or {}).get("pred", ""),
        "pred_scores": (row or {}).get("pred_scores", ""),
        "error": (row or {}).get("error", ""),
    }
    return rec


SLIDES = build_index()
BY_ID = {s["id"]: s for s in SLIDES}

# ------------------------------------------------- 組織クラスタ解析 (A-1〜3) ---
TISSUE_DIR = Path(CFG["tissue_dir"]) if CFG.get("tissue_dir") else ROOT / "tissue_analysis"
TISSUE: dict = {"clusters": [], "metrics": {}}


def load_tissue() -> None:
    TISSUE["clusters"] = []
    TISSUE["metrics"] = {}
    tj = TISSUE_DIR / "cluster_table.json"
    if tj.exists():
        try:
            TISSUE["clusters"] = json.loads(tj.read_text(encoding="utf-8")).get("clusters", [])
        except json.JSONDecodeError:
            pass
    tm = TISSUE_DIR / "slide_metrics.csv"
    if tm.exists():
        with tm.open(newline="", encoding="utf-8") as f:
            TISSUE["metrics"] = {r["stem"]: r for r in csv.DictReader(f)}


load_tissue()

# ------------------------------------------------- CONCH zero-shot (B) ---
CONCH: dict = {"classes": []}
_conch_npz: dict = {}        # results_dir 名 -> {stems, labels, counts}
_conch_metrics: dict = {}    # results_dir 名 -> {stem: csv row}


def load_conch() -> None:
    for cohort in COHORTS:
        rd = Path(cohort["results_dir"]).name
        res_dir = ROOT / cohort["results_dir"] / "conch_zeroshot"
        npz = CACHE / f"conch_{rd}.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        _conch_npz[rd] = {
            "stems": [str(s) for s in z["stems"]],
            "labels": z["labels"],
            "counts": z["counts"],
        }
        cj = res_dir / "classes.json"
        if cj.exists() and not CONCH["classes"]:
            CONCH["classes"] = json.loads(cj.read_text(encoding="utf-8"))["classes"]
        mcsv = res_dir / "slide_metrics.csv"
        if mcsv.exists():
            with mcsv.open(newline="", encoding="utf-8") as f:
                _conch_metrics[rd] = {r["stem"]: r for r in csv.DictReader(f)}


load_conch()

# ------------------------------------------------------- openslide cache ---
_slide_cache: "OrderedDict[int, openslide.OpenSlide]" = OrderedDict()
_slide_lock = threading.Lock()

# 低解像度側 (downsample > THUMB_FACTOR) のタイルは、この倍率のサムネイルから生成
THUMB_FACTOR = 8
_thumb_cache: "OrderedDict[int, Image.Image]" = OrderedDict()


def get_slide(sid: int) -> openslide.OpenSlide:
    rec = BY_ID.get(sid)
    if not rec or not rec["has_image"]:
        abort(404, "slide image not found")
    with _slide_lock:
        if sid in _slide_cache:
            s = _slide_cache.pop(sid)
            _slide_cache[sid] = s
            return s
    s = openslide.OpenSlide(str(ROOT / rec["path"]))
    with _slide_lock:
        _slide_cache[sid] = s
        while len(_slide_cache) > 4:
            _slide_cache.popitem(last=False)
    return s


def get_thumb(sid: int) -> Image.Image:
    """THUMB_FACTOR 倍率のサムネイル (ディスク+メモリキャッシュ)."""
    with _slide_lock:
        if sid in _thumb_cache:
            im = _thumb_cache.pop(sid)
            _thumb_cache[sid] = im
            return im
    rec = BY_ID[sid]
    s = get_slide(sid)
    p = ROOT / rec["path"]
    tag = hashlib.md5(f"{p}:{p.stat().st_mtime_ns}:tf{THUMB_FACTOR}".encode()).hexdigest()
    cache_jpg = CACHE / f"thumb_{tag}.jpg"
    if cache_jpg.exists():
        im = Image.open(cache_jpg)
        im.load()
    else:
        CACHE.mkdir(exist_ok=True)
        im = s.get_thumbnail(
            (max(1, s.dimensions[0] // THUMB_FACTOR),
             max(1, s.dimensions[1] // THUMB_FACTOR))
        ).convert("RGB")
        im.save(cache_jpg, quality=90)
    with _slide_lock:
        _thumb_cache[sid] = im
        while len(_thumb_cache) > 8:
            _thumb_cache.popitem(last=False)
    return im


# ----------------------------------------------------------------- api ---
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/api/slides")
def api_slides():
    return jsonify(
        {
            "cohorts": [c["name"] for c in COHORTS],
            "slides": SLIDES,
        }
    )


@app.get("/api/slide/<int:sid>")
def api_slide(sid: int):
    rec = BY_ID.get(sid)
    if rec is None:
        abort(404)
    info = dict(rec)
    if rec["has_image"]:
        s = get_slide(sid)
        w0, h0 = s.dimensions
        n_levels = 1 + max(0, math.ceil(math.log2(max(w0, h0))))
        info["levels"] = [
            {
                "level": i,
                "width": math.ceil(w0 / 2**i),
                "height": math.ceil(h0 / 2**i),
                "downsample": 2**i,
            }
            for i in range(n_levels)
        ]
        mpp = s.properties.get(openslide.PROPERTY_NAME_MPP_X)
        info["mpp"] = float(mpp) if mpp else None
    if rec["key"] in TISSUE["metrics"]:
        info["tissue"] = TISSUE["metrics"][rec["key"]]
    rd = Path(rec["h5"]).parent.name if rec["h5"] else ""
    if rd in _conch_metrics and rec["key"] in _conch_metrics[rd]:
        info["conch"] = _conch_metrics[rd][rec["key"]]
    return jsonify(info)


@app.get("/api/slide/<int:sid>/patches")
def api_patches(sid: int):
    """h5 のパッチ座標 +色を返す. mode=pca (デフォルト) or mode=cluster (組織クラスタ)."""
    rec = BY_ID.get(sid)
    if rec is None:
        abort(404)
    if not rec["has_h5"]:
        return jsonify({"coords": [], "colors": [], "patch_size": 0})

    h5p = ROOT / rec["h5"]
    tag = hashlib.md5(f"{h5p}:{h5p.stat().st_mtime_ns}".encode()).hexdigest()
    npz = CACHE / f"{tag}.npz"
    if npz.exists():
        z = np.load(npz)
        coords, colors, psize = z["coords"], z["colors"], int(z["psize"])
    else:
        with h5py.File(h5p, "r") as f:
            coords = f["coords"][:].astype(np.int64)
            psize = int(f["coords"].attrs.get("patch_size_level0", 512))
            feats = np.asarray(f["features"][:], dtype=np.float32)
        colors = _pca_colors(feats)
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(npz, coords=coords, colors=colors, psize=psize)

    mode = request.args.get("mode", "pca")
    if mode == "cluster":
        tc = CACHE / "tissue_clusters.npz"
        if not tc.exists():
            return jsonify({"coords": [], "colors": [], "patch_size": 0,
                            "note": "tissue_analysis が未実行です"})
        z = np.load(tc)
        stems = [str(s) for s in z["stems"]]
        if rec["key"] not in stems:
            return jsonify({"coords": [], "colors": [], "patch_size": 0,
                            "note": "このスライドは組織クラスタ解析の対象外です"})
        i = stems.index(rec["key"])
        ids = z["labels"][int(z["counts"][:i].sum()): int(z["counts"][:i + 1].sum())]
        palette = np.array([c["color"] for c in TISSUE["clusters"]], dtype=np.uint8)
        if len(ids) != len(coords):
            return jsonify({"coords": [], "colors": [], "patch_size": 0,
                            "note": "このスライドの patch 構成は組織クラスタ解析実行時と一致しないため、色分けできません"})
        colors = palette[ids]
    elif mode == "conch":
        rd = Path(rec["h5"]).parent.name
        if rd not in _conch_npz:
            return jsonify({"coords": [], "colors": [], "patch_size": 0,
                            "note": "CONCH zero-shot が未実行です (conch_zeroshot.py)"})
        d = _conch_npz[rd]
        if rec["key"] not in d["stems"]:
            return jsonify({"coords": [], "colors": [], "patch_size": 0,
                            "note": "このスライドは CONCH 解析の対象外です"})
        i = d["stems"].index(rec["key"])
        ids = d["labels"][int(d["counts"][:i].sum()): int(d["counts"][:i + 1].sum())]
        palette = np.array([c["color"] for c in CONCH["classes"]], dtype=np.uint8)
        if len(ids) != len(coords):
            return jsonify({"coords": [], "colors": [], "patch_size": 0,
                            "note": "CONCH 解析の patch 構成がこのビューワ行と一致しません"})
        colors = palette[ids]
        out_counts = [int((ids == k).sum()) for k in range(len(CONCH["classes"]))]
        result = {"patch_size": psize, "coords": coords.tolist(),
                  "colors": colors.tolist(), "counts": out_counts}
        return jsonify(result)
    return jsonify(
        {
            "patch_size": psize,
            "coords": coords.tolist(),
            "colors": colors.tolist(),
        }
    )


@app.get("/api/tissue")
def api_tissue():
    return jsonify(TISSUE)


@app.get("/api/conch")
def api_conch():
    return jsonify(CONCH)


def _pca_colors(feats: np.ndarray) -> np.ndarray:
    """768次元特徴量を PCA(2成分) で 'hot'  colormap 風 RGB に."""
    f = feats - feats.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(f, full_matrices=False)
        p = f @ vt[:2].T
    except np.linalg.LinAlgError:
        p = np.zeros((len(f), 2), dtype=np.float32)
    ang = np.arctan2(p[:, 1], p[:, 0])
    r = (p[:, 0] ** 2 + p[:, 1] ** 2) ** 0.5
    t = 0.5 * (ang / math.pi + 1.0) * 0.85 + 0.15 * (
        r.argsort().argsort() / max(len(r) - 1, 1)
    )
    t = np.clip(t, 0, 1)
    hot = np.stack(
        [
            np.clip(3 * t, 0, 1),
            np.clip(3 * t - 1, 0, 1),
            np.clip(3 * t - 2, 0, 1),
        ],
        axis=1,
    )
    return (hot * 255).astype(np.uint8)


@app.get("/slide/<int:sid>/tiles/<int:level>/<int:x>_<int:y>.png")
def tile(sid: int, level: int, x: int, y: int):
    s = get_slide(sid)
    w0, h0 = s.dimensions
    f = 2**level
    if f > max(w0, h0) * 2 or x < 0 or y < 0:
        abort(404)
    ox, oy = x * TILE * f, y * TILE * f
    if ox >= w0 or oy >= h0:
        abort(404)
    tw = min(TILE, math.ceil((w0 - ox) / f))
    th = min(TILE, math.ceil((h0 - oy) / f))
    try:
        if f <= THUMB_FACTOR:
            L = s.get_best_level_for_downsample(f)
            # read_region: location/size は level0 座標系。出力を (tw,th) に縮小
            img = s.read_region((ox, oy), L, (tw * f, th * f)).convert("RGB")
            if f > 1:
                img = img.resize((tw, th), Image.BILINEAR)
        else:
            base = get_thumb(sid)
            sf = f // THUMB_FACTOR
            tx, ty = x * TILE * sf, y * TILE * sf
            box = (
                min(tx, base.width),
                min(ty, base.height),
                min(base.width, tx + tw * sf),
                min(base.height, ty + th * sf),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                abort(404)
            img = base.crop(box).resize((tw, th), Image.BILINEAR)
    except Exception:
        app.logger.exception("tile error")
        abort(500)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return Response(buf.getvalue(), mimetype="image/jpeg")


@app.get("/api/report/<cohort>")
def api_report(cohort: str):
    for c in COHORTS:
        if c["name"] == cohort:
            rp = ROOT / c["results_dir"] / "report.html"
            if rp.exists():
                return send_from_directory(rp.parent, rp.name)
    abort(404)


@app.get("/reports/<path:name>")
def download_report(name: str):
    """reports/ の生成レポートをダウンロード (単一 HTML など)."""
    base = ROOT / "reports"
    rp = (base / name).resolve()
    if base.resolve() not in rp.parents or not rp.is_file():
        abort(404)
    return send_from_directory(rp.parent, rp.name, as_attachment=True)


if __name__ == "__main__":
    n_img = sum(1 for s in SLIDES if s["has_image"])
    n_res = sum(1 for s in SLIDES if s["has_h5"])
    print(f"slides: {len(SLIDES)}  (image: {n_img}, h5: {n_res})")
    app.run(host="0.0.0.0", port=8700, threaded=True)
