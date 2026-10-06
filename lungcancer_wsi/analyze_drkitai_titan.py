#!/usr/bin/env python3
"""
DrKitai TITAN 順次解析スクリプト
================================
data/DrKitai フォルダ内の .ndpi を1枚ずつ順次処理し、
TITAN slide embedding を抽出します。

パイプライン (TITAN公式に準拠):
  1. OpenSlide で WSI を開く (20X相当, level0座標系)
  2. 組織マスク (thumbnail + Otsu) で背景除外しタイル座標を決定
  3. CONCHv1.5 (titan.return_conch()) でパッチ特徴量抽出
  4. TITAN.encode_slide_from_patch_features() でスライド埋め込み
  5. results/ に .pt / .h5 + summary.csv を保存 (レジューム対応)

使い方:
  # まず一覧だけ確認 (重い依存不要, 標準ライブラリのみで動作)
  python3 analyze_drkitai_titan.py --dry-run

  # 環境構築 (Ubuntu/WSL例, OpenSlideシステムライブラリが必要):
  #   sudo apt update && sudo apt install -y openslide-tools python3-venv
  #   python3 -m venv .venv && source .venv/bin/activate
  #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  #   pip install transformers"<5" huggingface_hub h5py openslide-python pillow numpy pandas tqdm
  #   huggingface-cli login   # MahmoodLab/TITAN は gated のため事前にAccess申請
  #
  # 先頭2枚だけテスト:
  #   python3 analyze_drkitai_titan.py --limit 2
  # 全件実行:
  #   python3 analyze_drkitai_titan.py
  # GPU指定・上書き:
  #   python3 analyze_drkitai_titan.py --device cuda --overwrite

対象データ例:
  data/DrKitai/TITAN検討用 HE/G12C-1_HE - 2025-02-25 10.32.55.ndpi (59 files, ~18GB)
  ファイル名 prefix がラベル: G12C / other / wild (+ wild-6#12 のような枝番あり)

TITAN: https://github.com/petadimensionlab/TITAN (fork of mahmoodlab/TITAN)
"""

from __future__ import annotations

# user-space OpenSlide (sudo不要) を自動でLD pathに載せる
import os as _os
for _p in (f"{_os.environ.get('HOME','')}/.local/openslide/usr/lib/x86_64-linux-gnu",):
    if _os.path.isdir(_p) and _p not in _os.environ.get("LD_LIBRARY_PATH", ""):
        _os.environ["LD_LIBRARY_PATH"] = _p + ":" + _os.environ.get("LD_LIBRARY_PATH", "")

import argparse
import csv
import logging
import re
import sys
import time
import traceback
from pathlib import Path

# 肺癌KRASゼロショット用プロンプト (TITAN zero_shot_classifier用)
LUNG_PROMPTS = {
    "G12C": [
        "lung adenocarcinoma with KRAS G12C mutation.",
        "KRAS G12C mutant lung cancer histology.",
    ],
    "other": [
        "lung adenocarcinoma with non-G12C KRAS mutation.",
        "KRAS non-G12C mutant lung cancer histology.",
    ],
    "wild": [
        "KRAS wild-type lung adenocarcinoma.",
        "lung cancer without KRAS mutation histology.",
    ],
}

# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
DEFAULT_INPUT = Path(__file__).parent / "data" / "DrKitai"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "DrKitai_TITAN"
SUPPORTED_EXTS = {".ndpi", ".svs", ".tif", ".tiff", ".mrxs", ".scn"}

# 20X相当でのタイル間隔 (level0 px)。40Xスキャンなら1024, 20Xなら512。
# NDPIは機種・倍率で異なるため、OpenSlideの倍率を読んで自動切替する。
PATCH_SIZE_20X = 512
THUMBNAIL_MAX = 2048  # 組織検出用サムネイル長辺
TISSUE_SAT_THRESH = 20  # HSV-S閾値 (低いほど広く拾う)


def parse_label(filename: str, label_map: dict[str, str] | None = None) -> str:
    """ラベル推定。label_map (prefix->label) があれば最長一致で優先。"""
    stem = Path(filename).stem
    if label_map:
        keys = [k for k in label_map if stem.startswith(k)]
        if keys:
            return label_map[max(keys, key=len)]
    base = filename.lower()
    if base.startswith("g12c"):
        return "G12C"
    if base.startswith("other"):
        return "other"
    if base.startswith("wild"):
        return "wild"
    return "unknown"


def load_label_map(path: Path | None) -> dict[str, str]:
    """JSON (prefix->label) を読込。Noneなら空。"""
    if not path:
        return {}
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_slides(input_dir: Path) -> list[Path]:
    """再帰的にWSIファイルを列挙 (ソート済み)。"""
    files: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            # macOSの ._.リソースフォークや隠しファイルを除外
            if p.name.startswith("._") or p.name.startswith("."):
                continue
            files.append(p)
    return sorted(files)


# ----------------------------------------------------------------------------
# 重い処理 (遅延import: dry-run時はtorch/openslide不要)
# ----------------------------------------------------------------------------
def extract_slide_embedding(
    slide_path: Path,
    out_dir: Path,
    model_bundle: dict,
    args,
    logger: logging.Logger,
) -> dict:
    """1スライド分の処理。成功時はembedding path等を返す。例外は呼び出し側で記録。"""
    import numpy as np
    import torch

    t0 = time.time()
    titan = model_bundle["titan"]
    conch = model_bundle["conch"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]
    amp_dtype = model_bundle["amp_dtype"]

    # -- OpenSlideで開く -----------------------------------------------------
    try:
        import openslide
    except ImportError as e:
        raise RuntimeError(
            "openslide-python が未インストールです。"
            " sudo apt install -y openslide-tools && pip install openslide-python"
        ) from e

    from PIL import Image

    slide = openslide.OpenSlide(str(slide_path))
    w0, h0 = slide.dimensions
    # 対物倍率の推定 (なければ20X扱い)
    try:
        magnification = float(
            slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 20)
        )
    except (ValueError, TypeError):
        magnification = 20.0
    # level0間隔: 40X->1024, 20X->512 (TITAN docs準拠)。level0=スキャン倍率。
    # 20X相当512pxを切り出す: read = 512 * (mag/20)
    patch_size_lv0 = int(round(512 * (magnification / 20.0))) if magnification else 512
    # 実際に切り出すサイズ (20X換算512px相当)
    read_size = int(round(512 * (magnification / 20.0))) if magnification else 512

    logger.info(
        f"  size_lv0={w0}x{h0} mag~{magnification} patch_size_lv0={patch_size_lv0} read={read_size}px"
    )

    # -- 組織マスク (サムネイル + 彩度) ---------------------------------------
    thumb_w = THUMBNAIL_MAX if w0 >= h0 else int(w0 * THUMBNAIL_MAX / h0)
    thumb_h = int(h0 * THUMBNAIL_MAX / w0) if w0 >= h0 else THUMBNAIL_MAX
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    import numpy as _np  # local alias

    hsv = _np.array(thumb.convert("HSV"))
    saturation = hsv[:, :, 1]
    tissue_mask = saturation > TISSUE_SAT_THRESH
    sx = w0 / thumb.size[0]
    sy = h0 / thumb.size[1]

    # -- グリッド走査 --------------------------------------------------------
    step = read_size
    coords: list[list[int]] = []
    for y in range(0, h0 - read_size + 1, step):
        ty = int((y + read_size / 2) / sy)
        if ty >= tissue_mask.shape[0]:
            continue
        for x in range(0, w0 - read_size + 1, step):
            tx = int((x + read_size / 2) / sx)
            if tx >= tissue_mask.shape[1]:
                continue
            if not tissue_mask[ty, tx]:
                continue
            coords.append([x, y])

    if not coords:
        raise RuntimeError("組織タイルが0件でした (TISSUE_SAT_THRESHを下げて再試行)")

    # 全組織タイルから均等サンプリング (先頭偏りを排除しスライド全体をカバー)
    if len(coords) > args.max_patches:
        idx = _np.linspace(0, len(coords) - 1, args.max_patches).astype(int)
        coords = [coords[i] for i in idx]

    # -- パッチ特徴量 (CONCHv1.5) --------------------------------------------
    feats: list = []
    coords_kept: list[list[int]] = []
    batch_imgs = []
    batch_coords = []
    batch_size = args.batch_size

    def _flush():
        if not batch_imgs:
            return
        import torch as _torch

        with _torch.inference_mode():
            inp = _torch.stack([transform(img) for img in batch_imgs]).to(device)
            with _torch.autocast(device_type=device.type, dtype=amp_dtype):
                # このforkのCONCHは EncoderWithAttentionalPooler: forward(x)で特徴量
                # (旧CONCHの encode_image(proj_contrast...) APIではない)
                if hasattr(conch, "encode_image"):
                    f = conch.encode_image(inp, proj_contrast=False, normalize=False)
                else:
                    f = conch(inp)
        feats.append(f.float().cpu())
        coords_kept.extend(batch_coords)
        batch_imgs.clear()
        batch_coords.clear()

    for cx, cy in coords:
        try:
            patch = slide.read_region((cx, cy), 0, (read_size, read_size)).convert("RGB")
        except Exception:
            continue
        if read_size != 512:
            patch = patch.resize((512, 512), Image.BILINEAR)
        batch_imgs.append(patch)
        batch_coords.append([cx, cy])
        if len(batch_imgs) >= batch_size:
            _flush()
    _flush()
    slide.close()

    if not feats:
        raise RuntimeError("パッチ特徴量が0件でした")

    import torch as _torch

    features = _torch.cat(feats, dim=0)  # (N, D)
    coords_t = _torch.tensor(coords_kept, dtype=_torch.long)

    # -- TITAN slide embedding ----------------------------------------------
    with _torch.inference_mode():
        with _torch.autocast(device_type=device.type, dtype=amp_dtype):
            slide_emb = titan.encode_slide_from_patch_features(
                features.to(device), coords_t.to(device), patch_size_lv0
            )
    emb = slide_emb.detach().float().cpu().squeeze()  # (D,)

    # -- 保存 ----------------------------------------------------------------
    stem = re.sub(r"\s+", "_", slide_path.stem)  # 空白対策
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{stem}.pt"
    _torch.save({"embedding": emb, "source": str(slide_path), "n_patches": len(coords_kept)}, pt_path)

    # h5 (TITAN demo形式と互換: features/coords/patch_size_level0)
    h5_path = out_dir / f"{stem}.h5"
    try:
        import h5py

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("features", data=features.numpy())
            ds = f.create_dataset("coords", data=coords_kept)
            ds.attrs["patch_size_level0"] = patch_size_lv0
    except Exception as e:
        logger.warning(f"  h5保存スキップ: {e}")
        h5_path = None

    dt = time.time() - t0
    return {
        "n_patches": len(coords_kept),
        "embedding_dim": int(emb.numel()),
        "pt_path": str(pt_path),
        "h5_path": str(h5_path) if h5_path else "",
        "seconds": round(dt, 1),
    }


def load_models(checkpoint: str | None, device_str: str, logger: logging.Logger) -> dict:
    """TITAN + CONCH をロード。CUDA/MPS/CPU自動選択。"""
    import torch
    from transformers import AutoModel

    # TITAN fork の device ユーティリティがあれば使う
    try:
        sys.path.insert(0, str(Path(__file__).parent / "TITAN"))
        from titan.utils import amp_dtype as _amp, get_device as _get

        device = _get() if device_str == "auto" else torch.device(device_str)
        amp = _amp(device)
    except Exception:
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                try:
                    mps = torch.backends.mps
                    device = torch.device("mps") if mps.is_available() else torch.device("cpu")
                except AttributeError:
                    device = torch.device("cpu")
        else:
            device = torch.device(device_str)
        amp = torch.bfloat16 if device.type == "cuda" else torch.float16

    ckpt = checkpoint or "MahmoodLab/TITAN"
    logger.info(f"TITANロード: {ckpt} -> {device} (gated: 要HFログイン・Access承認)")
    titan = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
    titan = titan.to(device).eval()
    logger.info("CONCHv1.5 パッチエンコーダ取得中...")
    conch, transform = titan.return_conch()
    conch = conch.to(device).eval()
    return {"titan": titan, "conch": conch, "transform": transform, "device": device, "amp_dtype": amp}


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="DrKitai WSIをTITANで順次解析")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力フォルダ (default: data/DrKitai)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="出力フォルダ")
    ap.add_argument("--checkpoint", default=None, help="TITAN ckpt (default: MahmoodLab/TITAN)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (+mps可ならmps)")
    ap.add_argument("--max-patches", type=int, default=2000, help="1スライド上限パッチ数 (default: 2000)")
    ap.add_argument("--batch-size", type=int, default=32, help="CONCH batch (default: 32)")
    ap.add_argument("--limit", type=int, default=0, help="先頭N件のみ (0=全件, default: 0)")
    ap.add_argument("--overwrite", action="store_true", help="既存.ptを再計算")
    ap.add_argument("--dry-run", action="store_true", help="一覧表示のみ (torch不要)")
    ap.add_argument("--zeroshot", action="store_true", help="G12C/other/wildゼロショット分類まで実行")
    ap.add_argument("--label-map", type=Path, default=None, help="prefix->label のJSON (例: data/DrHatanaka/labels.json)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    label_map = load_label_map(args.label_map)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("drkitai-titan")

    if not args.input.exists():
        logger.error(f"入力フォルダがありません: {args.input}")
        return 2

    slides = find_slides(args.input)
    if args.limit and args.limit > 0:
        slides = slides[: args.limit]
    logger.info(f"対象: {len(slides)} files in {args.input}")
    for i, s in enumerate(slides, 1):
        try:
            size_mb = s.stat().st_size / 1e6
        except OSError:
            size_mb = -1
        logger.info(f"  [{i:02d}] {s.relative_to(args.input)} label={parse_label(s.name, label_map)} {size_mb:.0f}MB")

    if args.dry_run:
        print(f"\nDRY-RUN: {len(slides)} slides (no compute). Remove --dry-run to run TITAN.")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.csv"

    # レジューム: 既存summary読込
    done: dict[str, dict] = {}
    if summary_path.exists() and not args.overwrite:
        try:
            with open(summary_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done[row["filename"]] = row
            logger.info(f"既存summary読込: {len(done)}件 (成功分はスキップ)")
        except Exception as e:
            logger.warning(f"summary読込失敗 (無視): {e}")

    try:
        bundle = load_models(args.checkpoint, args.device, logger)
    except Exception as e:
        logger.error(f"モデルロード失敗: {e}")
        logger.error("-> huggingface-cli login 済みか、MahmoodLab/TITANへのAccess承認を確認してください。")
        traceback.print_exc()
        return 3

    # VRAMプレフライト (Strata占有時は警告)
    try:
        import torch as _t

        if _t.cuda.is_available():
            free, total = _t.cuda.mem_get_info()
            logger.info(f"VRAM free={free/1e9:.1f}GB / total={total/1e9:.1f}GB")
            if free < 8e9:
                logger.warning(
                    "VRAM残量<8GB: Windows側Strata(q2_0)が約23GB占有中と推定。 "
                    "TITAN+CONCHでOOMの可能性。Strata停止か --device cpu を検討。"
                )
    except Exception:
        pass

    # ゼロショット分類器 (任意)
    zs_classifier = None
    zs_classes: list[str] = []
    if args.zeroshot:
        try:
            import torch as _t

            from titan.utils import TEMPLATES

            zs_classes = sorted(LUNG_PROMPTS.keys())
            prompts = [LUNG_PROMPTS[c] for c in zs_classes]
            with _t.inference_mode():
                with _t.autocast(device_type=bundle["device"].type, dtype=bundle["amp_dtype"]):
                    zs_classifier = bundle["titan"].zero_shot_classifier(
                        prompts, TEMPLATES, device=bundle["device"]
                    )
            logger.info(f"ゼロショット分類器作成: {zs_classes}")
        except Exception as e:
            logger.error(f"ゼロショット分類器作成失敗 (埋め込みのみ継続): {e}")

    fieldnames = [
        "idx", "filename", "relative_path", "label", "size_mb",
        "status", "n_patches", "embedding_dim", "pt_path", "h5_path",
        "pred", "pred_scores", "seconds", "error",
    ]
    # 追記モードで1件ずつflush (中断耐性)
    write_header = not summary_path.exists() or args.overwrite
    fsum = open(summary_path, "w" if write_header else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fsum, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        fsum.flush()

    n_ok = n_skip = n_fail = 0
    for i, s in enumerate(slides, 1):
        rel = str(s.relative_to(args.input))
        prev = done.get(s.name)
        stem = re.sub(r"\s+", "_", s.stem)
        pt_exists = (args.output / f"{stem}.pt").exists()
        if not args.overwrite and (pt_exists or (prev and prev.get("status") == "ok")):
            logger.info(f"[{i}/{len(slides)}] SKIP {s.name}")
            n_skip += 1
            continue

        logger.info(f"[{i}/{len(slides)}] START {s.name} label={parse_label(s.name, label_map)}")
        try:
            size_mb = round(s.stat().st_size / 1e6, 1)
        except OSError:
            size_mb = -1
        try:
            r = extract_slide_embedding(s, args.output, bundle, args, logger)
            pred, scores_str = "", ""
            if zs_classifier is not None:
                try:
                    import torch as _t

                    d = _t.load(r["pt_path"], map_location=bundle["device"])
                    emb = d["embedding"] if isinstance(d, dict) else d
                    with _t.inference_mode():
                        with _t.autocast(device_type=bundle["device"].type, dtype=bundle["amp_dtype"]):
                            sc = bundle["titan"].zero_shot(
                                emb.to(bundle["device"]), zs_classifier
                            )
                    sc = sc.detach().float().cpu().flatten().tolist()
                    pred = zs_classes[int(__import__("numpy").argmax(sc))]
                    scores_str = "|".join(f"{c}:{v:.3f}" for c, v in zip(zs_classes, sc))
                    logger.info(f"  zeroshot pred={pred} [{scores_str}] true={parse_label(s.name, label_map)}")
                except Exception as ze:
                    logger.warning(f"  zeroshot失敗: {ze}")
            writer.writerow({
                "idx": i, "filename": s.name, "relative_path": rel,
                "label": parse_label(s.name, label_map), "size_mb": size_mb,
                "status": "ok", "n_patches": r["n_patches"],
                "embedding_dim": r["embedding_dim"], "pt_path": r["pt_path"],
                "h5_path": r["h5_path"], "pred": pred, "pred_scores": scores_str,
                "seconds": r["seconds"], "error": "",
            })
            n_ok += 1
            logger.info(f"[{i}/{len(slides)}] DONE {s.name} patches={r['n_patches']} {r['seconds']}s")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:500]
            logger.error(f"[{i}/{len(slides)}] FAIL {s.name}: {msg}")
            traceback.print_exc()
            writer.writerow({
                "idx": i, "filename": s.name, "relative_path": rel,
                "label": parse_label(s.name, label_map), "size_mb": size_mb,
                "status": "fail", "n_patches": "", "embedding_dim": "",
                "pt_path": "", "h5_path": "", "pred": "", "pred_scores": "",
                "seconds": "", "error": msg,
            })
            n_fail += 1
        fsum.flush()

    fsum.close()
    logger.info(f"完了: ok={n_ok} skip={n_skip} fail={n_fail} -> {summary_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
