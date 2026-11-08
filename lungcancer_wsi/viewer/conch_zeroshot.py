#!/usr/bin/env python3
"""CONCH (v1.0) zero-shot 組織分類 (B).

h5 に保存済みのパッチ座標 (A解析と同じグリッド) から NDPI を再切り出しし,
CONCH v1.0 (ViT-B/16, 画像/テキスト 512次元) で推論してテキスト埋め込みと
zero-shot 分類する.

注: results の h5 に保存された 768次元特徴は TITAN 同梱の CONCH **v1.5**
(ViT-L vision-only) の出力で, テキストエンコーダが公開されていないため
zero-shot には使えない → v1.0 でパッチを再推論する.

クラス: LUAD / LUSC / リンパ球 / 線維間質 / 肺胞腔 / 壊死 / 血液
テンプレ: CONCH 公式 nsclc_prompts (22 templates).

入力 : results/*/*.h5 (座標) + data/ の NDPI
出力 : results/<dir>/conch_zeroshot/  classes.json / slide_metrics.csv / report.md
       .viewer_cache/conch_<dir>.npz   (ビューワの mode=conch 用)

必要 : CONCH checkpoint  (HF MahmoodLab/CONCH, gated → models/conch_v1/pytorch_model.bin)
実行 : LD_LIBRARY_PATH=$HOME/.local/openslide/usr/lib/aarch64-linux-gnu \
       .venv-gpu/bin/python titan_viewer/conch_zeroshot.py [--dirs ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cohorts as cohorts_cfg  # noqa: E402

CFG = cohorts_cfg.load()
ROOT = Path(CFG["_root"])
RES = ROOT / "results"
CACHE = ROOT / ".viewer_cache"
DEFAULT_WEIGHTS = ROOT / "models/conch_v1/pytorch_model.bin"
CONCH_REPO = Path("/tmp/opencode/CONCH")  # プロンプトテンプレの取得元
ADJ_RADIUS_UM = 350.0

# results ディレクトリ名 -> 画像ディレクトリ (cohorts.json より)
COHORT_DATA = {Path(c["results_dir"]).name: c["data_dir"] for c in CFG["cohorts"]}

# クラス定義 (id, 表示名JA, コンマ区切り classnames)
CLASSES = [
    ("LUAD",   "肺腺癌",
     ["adenocarcinoma", "lung adenocarcinoma", "adenocarcinoma of the lung",
      "LUAD", "adenocarcinoma cells", "malignant glandular epithelium"]),
    ("LUSC",   "扁平上皮癌",
     ["squamous cell carcinoma", "lung squamous cell carcinoma",
      "squamous cell carcinoma of the lung", "LUSC",
      "squamous carcinoma cells", "keratinizing squamous cell carcinoma"]),
    ("LYMPH",  "リンパ球/炎症細胞",
     ["lymphocytes", "lymphocytic infiltrate", "lymphoid aggregate",
      "immune cell infiltrate", "inflammatory cells"]),
    ("STROMA", "線維間質/コラーゲン",
     ["fibrous stroma", "collagenous stroma", "fibrosis",
      "desmoplastic stroma", "scar tissue"]),
    ("ALV",    "肺胞腔/背景",
     ["empty background", "alveolar air space", "alveolar spaces",
      "acellular space", "lung alveoli"]),
    ("NECR",   "壊死",
     ["necrosis", "necrotic debris", "cell debris", "dead tissue"]),
    ("BLD",    "血液",
     ["blood", "hemorrhage", "erythrocytes", "red blood cells",
      "vascular lumen with blood"]),
]
COLORS = {"LUAD": (212, 58, 58), "LUSC": (163, 53, 43), "LYMPH": (47, 111, 212),
          "STROMA": (58, 165, 91), "ALV": (204, 204, 204), "NECR": (107, 74, 43),
          "BLD": (139, 31, 75)}
TUMOR_IDS = {"LUAD", "LUSC"}


def load_templates() -> list[str]:
    p = CONCH_REPO / "prompts/nsclc_prompts_all_per_class.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["0"]["templates"]
    return ["CLASSNAME.", "a photomicrograph showing CLASSNAME.",
            "a photomicrograph of CLASSNAME.", "there is CLASSNAME.",
            "this is a patch of CLASSNAME."]


def build_model(weights: Path):
    """open_clip_custom で conch_ViT-B-16 を構築し checkpoint を流し込む.

    checkpoint 側の attn_pool_contrast 次元が json (512) と違う場合
    (将来 v1.5 full 等が使われた場合) は形状を復元してロードする.
    """
    import importlib.util
    import torch.nn as nn
    from conch.open_clip_custom.coca_model import CoCa
    from conch.open_clip_custom.factory import read_state_dict
    from conch.open_clip_custom.transformer import AttentionalPooler

    loc = list(importlib.util.find_spec('conch').submodule_search_locations)[0]
    cfg_path = Path(loc) / "open_clip_custom/model_configs/conch_ViT-B-16.json"
    cfg = json.loads(cfg_path.read_text())
    # 公開 checkpoint から caption decoder 重みは削除済み → 構築しない (高速化)
    cfg["multimodal_cfg"]["layers"] = 0
    cfg.pop("custom_text", None)

    sd = read_state_dict(str(weights))
    cdim = sd["visual.ln_contrast.weight"].shape[0]
    edim = cfg["embed_dim"]                                   # 512 (contrast 出力)
    model = CoCa(**cfg)
    if cdim != edim:
        v = model.visual
        v.attn_pool_contrast = AttentionalPooler(
            d_model=cdim, context_dim=v.trunk.num_features,
            n_head=cfg["vision_cfg"]["attn_pooler_heads"],
            n_queries=cfg["vision_cfg"]["n_queries_contrast"])
        v.ln_contrast = nn.LayerNorm(cdim)
        v.proj_contrast = nn.Parameter(
            (cdim ** -0.5) * torch.randn(cdim, edim))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("text_decoder")]
    if bad:
        raise RuntimeError(f"checkpoint に無い必須重み: {bad[:10]}")
    print(f"  contrast dim: {cdim} -> {edim}, unexpected={len(unexpected)}")
    return model


def _tokenize(tok, texts):
    """conch tokenize 相当 (新版 transformers では batch_encode_plus が無い)."""
    import torch.nn.functional as FF
    try:
        enc = tok.batch_encode_plus(texts, max_length=127, add_special_tokens=True,
                                    return_token_type_ids=False, truncation=True,
                                    padding="max_length", return_tensors="pt")
    except AttributeError:
        enc = tok(texts, max_length=127, add_special_tokens=True, truncation=True,
                  padding="max_length", return_tensors="pt")
    return FF.pad(enc["input_ids"], (0, 1), value=tok.pad_token_id)


def build_text_weights(model, device) -> torch.Tensor:
    from conch.open_clip_custom import get_tokenizer
    tok = get_tokenizer()
    templates = load_templates()
    weights = []
    with torch.no_grad():
        for _, _, names in CLASSES:
            texts = [t.replace("CLASSNAME", c) for c in names for t in templates]
            embs = []
            for i in range(0, len(texts), 64):
                ids = _tokenize(tok, texts[i:i + 64]).to(device)
                e = model.encode_text(ids)  # normalize=True 済み
                embs.append(e)
            e = torch.cat(embs).mean(0)
            weights.append(e / e.norm())
    return torch.stack(weights, dim=1)  # (512, C)


def make_transform():
    from conch.open_clip_custom.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    from conch.open_clip_custom.transform import image_transform
    # factory.py と同じ前処理: 448 resize+centercrop, OpenAI CLIP 正規化
    return image_transform(image_size=448, mean=OPENAI_DATASET_MEAN,
                           std=OPENAI_DATASET_STD)


def stem_key(name: str) -> str:
    return Path(name).stem.replace(" ", "_")


def classify_slide(ndpi: Path, coords: np.ndarray, model, transform,
                   W: torch.Tensor, scale: torch.Tensor, dev,
                   batch: int = 96, amp=torch.bfloat16):
    """NDPI から coords のパッチを読み CONCH 推論 → (labels, probs, top_logits)."""
    slide = openslide.OpenSlide(str(ndpi))
    try:
        mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER,
                                         20) or 20)
    except (ValueError, TypeError):
        mag = 20.0
    read = int(round(512 * (mag / 20.0))) if mag else 512

    n = len(coords)
    labels = np.empty(n, dtype=np.uint8)
    probs = np.empty(n, dtype=np.float32)
    topk = {c: [] for c in range(W.shape[1])}
    try:
        for i in range(0, n, batch):
            imgs = []
            for x, y in coords[i:i + batch]:
                patch = slide.read_region((int(x), int(y)), 0, (read, read)).convert("RGB")
                if read != 512:
                    patch = patch.resize((512, 512), Image.BILINEAR)
                imgs.append(transform(patch))
            with torch.inference_mode():
                inp = torch.stack(imgs).to(dev)
                with torch.autocast(device_type=dev.type, dtype=amp):
                    z = model.encode_image(inp, normalize=True)  # (B, 512)
                logits = scale * (z.float() @ W)                 # (B, C)
            p = torch.softmax(logits, dim=-1)
            lab = logits.argmax(-1)
            labels[i:i + batch] = lab.cpu().numpy()
            probs[i:i + batch] = p.gather(1, lab[:, None]).cpu().numpy().ravel()
            for c in range(W.shape[1]):
                topk[c].append(logits[:, c].cpu())
    finally:
        slide.close()
    top_logits = {c: torch.cat(v) for c, v in topk.items()}
    return labels, probs, top_logits


def slide_metrics(labels: np.ndarray, coords: np.ndarray, mpp: float) -> dict:
    ids = [c[0] for c in CLASSES]
    out = {"n": int(len(labels))}
    for i, cid in enumerate(ids):
        out[f"frac_{cid}"] = round(float((labels == i).mean()), 4)
    it = [ids.index(c) for c in TUMOR_IDS]
    il = ids.index("LYMPH")
    tumor = np.isin(labels, it)
    lymph = labels == il
    ctr = (coords + 256) * mpp
    tu, ly = ctr[tumor], ctr[lymph]
    med_d = adj_t2l = None
    if len(tu) and len(ly):
        d = np.sqrt(((ly[:, None, :] - tu[None, :, :]) ** 2).sum(-1))
        med_d = float(np.median(d.min(1)))
        adj_t2l = float((d.min(0) <= ADJ_RADIUS_UM).mean())
    ft = sum(out[f"frac_{c}"] for c in TUMOR_IDS)
    fl = out["frac_LYMPH"]
    if fl < 0.03:
        pattern = "cold (浸潤乏しい)"
    elif med_d is not None and med_d <= 500.0 and (adj_t2l or 0) >= 0.35:
        pattern = "inflamed (腫瘍内浸潤)"
    else:
        pattern = "excluded (排除型)"
    out.update(lymph_to_tumor_med_um=None if med_d is None else round(med_d, 1),
               tumor_adj_lymph=None if adj_t2l is None else round(adj_t2l, 3),
               frac_tumor=round(ft, 4), pattern=pattern)
    return out


def welch(a, b):
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    t = (ma - mb) / math.sqrt(va + vb)
    return ma, mb, t, math.erfc(abs(t) / math.sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--dirs", nargs="*", default=list(COHORT_DATA))
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="各 dirs で先頭 N スライドのみ (0=全件)")
    args = ap.parse_args()

    wp = Path(args.weights)
    if not wp.exists():
        raise SystemExit(f"checkpoint が見つかりません: {wp}\n"
                         "HF (MahmoodLab/CONCH, gated) からダウンロードしてから実行してください。")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}, weights: {wp}")
    model = build_model(wp).to(dev).eval()
    W = build_text_weights(model, dev)
    scale = model.logit_scale.exp().to(dev).detach()
    print(f"text classifier: {W.shape[1]} classes, logit_scale={float(scale):.2f}")
    transform = make_transform()

    # metadata.csv (mpp 等の補完用): cohorts.json の meta_csv があればそれを使う
    _mc = next((c["meta_csv"] for c in CFG["cohorts"] if c.get("meta_csv")), None)
    meta_csv = Path(_mc) if _mc else RES / "metadata.csv"
    meta = {}
    if meta_csv.exists():
        meta = {r["stem"].replace(" ", "_"): r
                for r in csv.DictReader(meta_csv.open(encoding="utf-8"))}

    ids = [c[0] for c in CLASSES]
    all_rows = []
    for d in args.dirs:
        res = RES / d
        h5s = sorted(res.glob("*.h5"))
        if args.limit:
            h5s = h5s[:args.limit]
        if not h5s:
            continue
        data_dir = ROOT / COHORT_DATA.get(d, "data")
        ndpi_idx = {stem_key(p.name): p for p in data_dir.glob("*.ndpi")
                    if not p.name.startswith("._")}
        out_dir = res / "conch_zeroshot"
        out_dir.mkdir(exist_ok=True)
        stems, labels_all, counts = [], [], []
        rows = []
        for h5p in h5s:
            st = h5p.stem
            with h5py.File(h5p, "r") as f:
                coords = f["coords"][:].astype(np.int64)
            ndpi = ndpi_idx.get(st)
            if ndpi is None:
                print(f"  SKIP {st}: NDPI が見つかりません ({data_dir})")
                continue
            lab, prob, top_logits = classify_slide(
                ndpi, coords, model, transform, W, scale, dev, batch=args.batch)
            stems.append(st)
            labels_all.append(lab)
            counts.append(len(lab))
            m = meta.get(st, {})
            mpp = float(m.get("mpp_x") or 0.4545)
            mt = slide_metrics(lab, coords, mpp)
            # スライドレベル LUAD vs LUSC (logits top10 平均, MI-Zero 風)
            tl10 = {c: float(top_logits[c].topk(min(10, len(top_logits[c]))).values.mean())
                    for c in range(len(CLASSES))}
            mt.update(case_id=m.get("case_id", ""), specimen=m.get("specimen", ""),
                      luad_logit=round(tl10[0], 2), lusc_logit=round(tl10[1], 2),
                      luad_prob=float(torch.softmax(torch.tensor([tl10[0], tl10[1]]), 0)[0]))
            rows.append({"stem": st, "cohort": d, **mt})
            print(f"  {st}: n={len(lab)} mean_conf={prob.mean():.2f} "
                  f"LUADp={mt['luad_prob']:.2f}")
        lab_all = np.concatenate(labels_all)
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(CACHE / f"conch_{d}.npz",
                            stems=np.array(stems), labels=lab_all,
                            counts=np.array(counts, dtype=np.int64),
                            ids=np.array(ids))
        cols = ["stem", "cohort", "case_id", "specimen", "n"] + \
               [f"frac_{c}" for c in ids] + \
               ["frac_tumor", "lymph_to_tumor_med_um", "tumor_adj_lymph",
                "luad_logit", "lusc_logit", "luad_prob", "pattern"]
        with (out_dir / "slide_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        (out_dir / "classes.json").write_text(json.dumps(
            {"classes": [{"id": i, "ja": j, "names": n, "color": list(COLORS[i])}
                         for i, j, n in CLASSES]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        all_rows.extend(rows)

        # ---- cohort 内レポート ----
        lines = [f"# CONCH zero-shot 組織分類 — {d}", ""]
        cnt = Counter(int(x) for x in lab_all)
        tot = len(lab_all)
        lines.append("## patch クラス構成")
        for i, cid in enumerate(ids):
            lines.append(f"- {cid}: {cnt[i]} ({100*cnt[i]/tot:.1f}%)")
        specs = sorted({r["specimen"] for r in rows if r["specimen"]})
        if specs:
            lines.append("\n## 検体群比較")
            for gi in range(len(specs)):
                for gj in range(gi + 1, len(specs)):
                    ra = [r for r in rows if r["specimen"] == specs[gi]]
                    rb = [r for r in rows if r["specimen"] == specs[gj]]
                    for key in ["frac_tumor", "frac_LYMPH", "frac_STROMA",
                                "tumor_adj_lymph", "lymph_to_tumor_med_um"]:
                        va = np.array([r[key] for r in ra if r[key] is not None], float)
                        vb = np.array([r[key] for r in rb if r[key] is not None], float)
                        if len(va) >= 3 and len(vb) >= 3:
                            ma, mb, t, p = welch(va, vb)
                            lines.append(f"- {key}: {specs[gi]} {ma:.3f}±{va.std(ddof=1):.3f}"
                                         f" (n={len(va)}) vs {specs[gj]} {mb:.3f}±{vb.std(ddof=1):.3f}"
                                         f" (n={len(vb)}), t={t:.2f}, p≈{p:.3f}")
        lines.append("\n## 浸潤パターンの内訳")
        for sp in specs or [""]:
            c = Counter(r["pattern"] for r in rows if r["specimen"] == sp)
            if c:
                lines.append(f"- {sp}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
        lines.append("\n## スライドレベル LUAD/LUSC (logit top10 平均)")
        for r in sorted(rows, key=lambda r: -r["luad_prob"]):
            lines.append(f"- {r['stem']}: LUAD p={r['luad_prob']:.2f} "
                         f"({r['luad_logit']:.1f} vs {r['lusc_logit']:.1f})")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n{d}: {len(rows)} slides -> {out_dir}")


if __name__ == "__main__":
    main()
