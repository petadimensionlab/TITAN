#!/usr/bin/env python3
"""A-1〜A-3: 対象コホート (デフォルト: cohorts.json の最初のコホート、
環境変数 TISSUE_COHORT で results ディレクトリ名を指定して切替) の
パッチ特徴クラスタリング → 組織領域の自動抽出 → 免疫浸潤度の定量と
浸潤パターン (inflamed / excluded / cold) 分類.

入力 : <results_dir>/*.h5  (CONCH 768次元パッチ特徴 + level0 座標)
       <results_dir>/metadata.csv (case_id, specimen, mpp_x)
出力 : <tissue_dir> (= cohorts.json の tissue_dir, 省略時 <results_dir>/tissue_analysis)
         cluster_table.json/csv  クラスタ統計 + 暫定組織ラベル
         slide_metrics.csv       スライドごとの組織構成・免疫浸潤指標
         report.md               要約
       .viewer_cache/tissue_clusters.npz  (ビューワのクラスター色分け用)

ラベル付けは H&E 染色脱畳出し (Ruifrok-Johnston) による色・核密度統計に基づく
暫定判定。クラスタ単位で付与され、各パッチの色は 0.45µm 相当のサムネイル
(downsample 8) から計算。要視覚検証。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cohorts as cohorts_cfg  # noqa: E402

CFG = cohorts_cfg.load()
ROOT = Path(CFG["_root"])
_want = os.environ.get("TISSUE_COHORT", "")
_c = next((c for c in CFG["cohorts"] if Path(c["results_dir"]).name == _want), CFG["cohorts"][0])
COHORT_NAME = _c["name"]
RES = Path(_c["results_dir"])
DATA = Path(_c["data_dir"])
CACHE = ROOT / ".viewer_cache"
OUT = Path(CFG.get("tissue_dir") or (RES / "tissue_analysis"))
THUMB_FACTOR = 8
K = 12
SEED = 0
ADJ_RADIUS_UM = 350.0  # パッチ一辺 ~230µm なので隣接~斜め隣相当

# Ruifrok & Johnston H&E 染色マトリクス (行 = 染色, 列 = RGB)
_STAIN = np.array(
    [[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]],
    dtype=np.float64,
)
_STAIN /= np.linalg.norm(_STAIN, axis=1, keepdims=True)
_STAIN_INV = np.linalg.inv(_STAIN)

CLASS_BASE = {
    "lymphoid": (59, 111, 212),   # 青: リンパ球濃厚候補
    "tumor":    (212, 69, 59),    # 赤: 腫瘍/上皮候補
    "stroma":   (59, 165, 91),    # 緑: 線維間質候補
    "empty":    (204, 204, 204),  # 灰: 肺胞腔/脂肪/空白候補
    "other":    (176, 122, 161),  # 紫: その他
}
CLASS_JA = {
    "lymphoid": "リンパ球濃厚候補",
    "tumor": "腫瘍/上皮候補",
    "stroma": "線維間質候補",
    "empty": "肺胞腔/空白候補",
    "other": "その他",
}


def stem_key(name: str) -> str:
    return Path(name).stem.replace(" ", "_")


def find_image(stem: str) -> Path | None:
    for p in DATA.glob("*"):
        if p.name.startswith("._"):
            continue
        if stem_key(p.name) == stem and p.suffix.lower() in {".ndpi", ".svs", ".tif", ".tiff"}:
            return p
    return None


def thumbnail(stem: str) -> Image.Image:
    """ビューワと同じタグのキャッシュサムネイル (downsample THUMB_FACTOR)."""
    p = find_image(stem)
    if p is None:
        raise FileNotFoundError(stem)
    tag = hashlib.md5(f"{p}:{p.stat().st_mtime_ns}:tf{THUMB_FACTOR}".encode()).hexdigest()
    jp = CACHE / f"thumb_{tag}.jpg"
    if not jp.exists():
        import openslide
        s = openslide.OpenSlide(str(p))
        im = s.get_thumbnail(
            (max(1, s.dimensions[0] // THUMB_FACTOR),
             max(1, s.dimensions[1] // THUMB_FACTOR))
        ).convert("RGB")
        CACHE.mkdir(exist_ok=True)
        im.save(jp, quality=90)
    im = Image.open(jp)
    im.load()
    return im.convert("RGB")


def patch_color_stats(img: Image.Image, x: int, y: int, ps: int) -> tuple:
    """パッチ領域の H&E 濃度統計. 返り値 (od_mean, h_mean, e_mean, h_var, h_frac)."""
    tx, ty = x // THUMB_FACTOR, y // THUMB_FACTOR
    tw = max(2, ps // THUMB_FACTOR)
    box = (tx, ty, min(img.width, tx + tw), min(img.height, ty + tw))
    if box[2] <= box[0] or box[3] <= box[1]:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    a = np.asarray(img.crop(box).resize((tw, tw), Image.BILINEAR), dtype=np.float64)
    od = -np.log10(np.clip(a, 1, 255) / 255.0)
    c = od.reshape(-1, 3) @ _STAIN_INV
    h, e = c[:, 0], c[:, 1]
    return (
        float(od.mean()), float(h.mean()), float(e.mean()),
        float(h.std()), float((h > 0.18).mean()),
    )


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 40, restarts: int = 5):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    n = len(x)
    for _ in range(restarts):
        idx = rng.choice(n, size=k, replace=False)
        cent = x[idx].copy()
        x2 = (x * x).sum(1)[:, None]
        for _ in range(iters):
            d = x2 - 2.0 * (x @ cent.T) + (cent * cent).sum(1)[None, :]
            lab = d.argmin(1)
            newc = np.stack([x[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
            shift = ((newc - cent) ** 2).sum()
            cent = newc
            if shift < 1e-6:
                break
        wcss = (((x - cent[lab]) ** 2).sum()).item()
        if wcss < best[0]:
            best = (wcss, lab.astype(np.uint8), cent)
    return best[1], best[2]


def main() -> None:
    mpath = RES / "metadata.csv"
    if not mpath.exists():
        _mc = next((c["meta_csv"] for c in CFG["cohorts"] if c.get("meta_csv")), None)
        mpath = Path(_mc) if _mc else mpath
    meta = {r["stem"].replace(" ", "_"): r
            for r in csv.DictReader(mpath.open(encoding="utf-8"))}
    scsv = RES / "summary.csv"
    if scsv.exists():
        for r in csv.DictReader(scsv.open(encoding="utf-8")):
            k = stem_key(r.get("filename", ""))
            if k in meta:
                meta[k]["label"] = r.get("label", "")
    stems = sorted(p.stem for p in RES.glob("*.h5"))
    print(f"slides: {len(stems)}")

    # ---- 特徴と座標の読み込み --------------------------------------------
    all_feat, all_ids, all_stats, all_coords, slide_span = [], [], [], [], []
    psize = 512
    for st in stems:
        with h5py.File(RES / f"{st}.h5", "r") as f:
            coords = f["coords"][:].astype(np.int64)
            psize = int(f["coords"].attrs.get("patch_size_level0", 512))
            feats = np.asarray(f["features"][:], dtype=np.float32)
        img = thumbnail(st)
        stt = np.stack([patch_color_stats(img, int(x), int(y), psize) for x, y in coords])
        all_feat.append(feats)
        all_coords.append(coords)
        all_stats.append(stt)
        all_ids.append(np.full(len(feats), stems.index(st), dtype=np.int32))
        slide_span.append((st, len(feats)))
    F = np.concatenate(all_feat)
    C = np.concatenate(all_coords)
    S = np.concatenate(all_stats)  # (N,5): od,h,e,hvar,hfrac
    SID = np.concatenate(all_ids)
    N = len(F)
    print(f"patches: {N}")

    # ---- クラスタリング ----------------------------------------------------
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    Fc = Fn - Fn.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Fc, full_matrices=False)
    X = Fc @ vt[:48].T
    lab, cent = kmeans(X, K, seed=SEED)
    print(f"kmeans done: K={K}")

    # ---- クラスタ統計と暫定ラベル -----------------------------------------
    stat_names = ["od", "h", "e", "h_var", "h_frac"]
    pct = {n: {q: np.percentile(S[:, i], q) for q in (20, 45, 60, 75)}
           for i, n in enumerate(stat_names)}
    table = []
    for j in range(K):
        m = lab == j
        mu = S[m].mean(0)
        row = {"cluster": j, "n": int(m.sum())}
        row.update({n: round(float(v), 4) for n, v in zip(stat_names, mu)})
        od, h, e, hvar, hf = mu
        if od < pct["od"][20]:
            cls = "empty"
        elif h >= pct["h"][75]:
            cls = "lymphoid"
        elif e >= pct["e"][60] and h <= pct["h"][45]:
            cls = "stroma"
        elif h >= pct["h"][45]:
            cls = "tumor"
        else:
            cls = "other"
        row["cls"] = cls
        row["cls_ja"] = CLASS_JA[cls]
        table.append(row)

    # クラスタごとの色 (クラス基色 + クラスタごとに明度変化)
    seen: Counter = Counter()
    for row in table:
        i = seen[row["cls"]]
        seen[row["cls"]] += 1
        base = np.array(CLASS_BASE[row["cls"]], dtype=float)
        blend = 0.0 if i == 0 else 0.18 * (1 if i % 2 else -1) * ((i + 1) // 2)
        col = np.clip(base + (255 - base) * blend if blend > 0 else base * (1 + blend), 0, 255)
        row["color"] = [int(v) for v in col]
    palette = np.array([r["color"] for r in table], dtype=np.uint8)
    cls_of = np.array([r["cls"] for r in table])

    # ---- スライド別指標 ----------------------------------------------------
    percase = {}
    for si, st in enumerate(stems):
        m = SID == si
        cl = cls_of[lab[m]]
        co = C[m]
        mpp = float(meta[st]["mpp_x"]) if st in meta and meta[st]["mpp_x"] else 0.4545
        frac = {c: float((cl == c).mean()) for c in CLASS_BASE}
        # 中心座標 (µm)
        ctr = (co + psize / 2) * mpp
        ly, tu = ctr[cl == "lymphoid"], ctr[cl == "tumor"]
        med_d = mix_l2t = adj_t2l = None
        if len(ly) and len(tu):
            d = np.sqrt(((ly[:, None, :] - tu[None, :, :]) ** 2).sum(-1))
            med_d = float(np.median(d.min(1)))
            mix_l2t = float((d.min(1) <= ADJ_RADIUS_UM).mean())
            adj_t2l = float((d.min(0) <= ADJ_RADIUS_UM).mean())
        if frac["lymphoid"] < 0.03:
            pattern = "cold (浸潤乏しい)"
        elif med_d is not None and med_d <= 500.0 and mix_l2t >= 0.35:
            pattern = "inflamed (腫瘍内浸潤)"
        else:
            pattern = "excluded (排除型)"
        percase[st] = {
            "specimen": meta.get(st, {}).get("specimen", ""),
            "case_id": meta.get(st, {}).get("case_id", ""),
            "label": meta.get(st, {}).get("label", ""),
            "n": int(m.sum()), **{f"frac_{c}": round(frac[c], 4) for c in CLASS_BASE},
            "lymph_to_tumor_med_um": None if med_d is None else round(med_d, 1),
            "lymph_adj_tumor": None if mix_l2t is None else round(mix_l2t, 3),
            "tumor_adj_lymph": None if adj_t2l is None else round(adj_t2l, 3),
            "pattern": pattern,
        }

    # ---- 保存 --------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cluster_table.json").write_text(
        json.dumps({"K": K, "seed": SEED, "adj_radius_um": ADJ_RADIUS_UM,
                    "thresholds": {n: {str(q): round(v, 4) for q, v in d.items()}
                                   for n, d in pct.items()},
                    "clusters": table}, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["cluster", "n", "cls", "cls_ja", "od", "h", "e", "h_var", "h_frac", "color"]
    with (OUT / "cluster_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: r[c] for c in cols})
    mcols = ["stem", "specimen", "case_id", "label", "n"] + \
            [f"frac_{c}" for c in CLASS_BASE] + \
            ["lymph_to_tumor_med_um", "lymph_adj_tumor", "tumor_adj_lymph", "pattern"]
    with (OUT / "slide_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mcols)
        w.writeheader()
        for st in stems:
            w.writerow({"stem": st, **percase[st]})
    CACHE.mkdir(exist_ok=True)
    counts = np.array([n for _, n in slide_span], dtype=np.int64)
    np.savez_compressed(CACHE / "tissue_clusters.npz",
                        stems=np.array(stems), labels=lab, counts=counts,
                        patch_size=np.int64(psize))

    # ---- 群体比較と報告 ----------------------------------------------------
    def group_rows(name):
        return [v for v in percase.values() if v["specimen"] == name]

    def welch(a, b):
        ma, mb = np.mean(a), np.mean(b)
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
        t = (ma - mb) / math.sqrt(va + vb)
        p = math.erfc(abs(t) / math.sqrt(2))  # 正規近似
        return ma, mb, t, p

    lines = [f"# {COHORT_NAME} コホート 組織クラスタ・免疫浸潤解析 (A-1〜A-3)\n",
             f"- 対象: {len(stems)} スライド / {N} パッチ (パッチ {psize}px level0)",
             f"- クラスタ数 K={K}, 色・核密度統計による暫定ラベル (要視覚検証)\n",
             "## クラスタ一覧"]
    for r in table:
        lines.append(f"- C{r['cluster']} [{r['cls_ja']}] n={r['n']} "
                     f"od={r['od']} h={r['h']} e={r['e']} h_var={r['h_var']}")
    specs_all = sorted({v["specimen"] for v in percase.values() if v["specimen"]})
    if len(specs_all) > 1:
        lines.append(f"\n## 検体群比較 ({specs_all[0]} vs {specs_all[1]})")
    else:
        lines.append("\n## 検体群比較")
    specs = specs_all
    ga = group_rows(specs[0]) if specs else []
    gb = group_rows(specs[1]) if len(specs) > 1 else []
    for key in ["frac_lymphoid", "frac_tumor", "frac_stroma", "tumor_adj_lymph",
                "lymph_to_tumor_med_um"]:
        va = np.array([v[key] for v in ga if v[key] is not None], dtype=float)
        vb = np.array([v[key] for v in gb if v[key] is not None], dtype=float)
        if len(va) >= 3 and len(vb) >= 3:
            ma, mb, t, p = welch(va, vb)
            lines.append(f"- {key}: {specs[0]} {ma:.3f}±{va.std(ddof=1):.3f} (n={len(va)}) vs "
                         f"{specs[1]} {mb:.3f}±{vb.std(ddof=1):.3f} (n={len(vb)}), t={t:.2f}, p≈{p:.3f}")
    lines.append("\n## 浸潤パターンの内訳")
    for grp, rows in ((specs[0] if specs else "", ga), (specs[1] if len(specs) > 1 else "", gb)):
        cnt = Counter(v["pattern"] for v in rows)
        lines.append(f"- {grp}: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
