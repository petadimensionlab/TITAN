#!/usr/bin/env python3
"""DrKitai TITAN embedding linear probe (層化5-fold CV, TITAN公式手順準拠)。

使い方: source .venv-titan/bin/activate && python linear_probe_drkitai.py
入力: results/DrKitai_TITAN/*.pt + summary.csv
出力: results/DrKitai_TITAN/linear_probe.json + 標準出力にmetrics
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).parent
DEFAULT_EMB_DIR = ROOT / "results" / "DrKitai_TITAN"

sys.path.insert(0, str(ROOT / "TITAN"))
from titan.utils import get_eval_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="TITAN embedding linear probe (層化5-fold CV)")
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR, help="summary.csv/*.pt のあるフォルダ")
    ap.add_argument("--summary", type=Path, default=None, help="summary.csv (default: <emb-dir>/summary.csv)")
    ap.add_argument("--out", type=Path, default=None, help="出力JSON (default: <emb-dir>/linear_probe.json)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    EMB_DIR = args.emb_dir
    SUMMARY = args.summary or EMB_DIR / "summary.csv"
    OUT = args.out or EMB_DIR / "linear_probe.json"

    rows = [r for r in csv.DictReader(open(SUMMARY, encoding="utf-8")) if r["status"] == "ok"]
    labels = sorted(set(r["label"] for r in rows))
    lab2id = {l: i for i, l in enumerate(labels)}
    X, y, names = [], [], []
    for r in rows:
        p = Path(r["pt_path"])
        if not p.exists():
            p = EMB_DIR / (Path(r["filename"]).stem.replace(" ", "_") + ".pt")
        d = torch.load(p, map_location="cpu", weights_only=True)
        e = d["embedding"] if isinstance(d, dict) else d
        X.append(np.asarray(e, dtype=np.float64).ravel())
        y.append(lab2id[r["label"]])
        names.append(r["filename"])
    X = Normalizer(norm="l2").fit_transform(np.stack(X))
    y = np.array(y)
    print(f"n={len(y)} dim={X.shape[1]} classes={labels} dist={Counter(y)}")
    # wild-6#12/#16 同一症例疑いを注意喚起
    dup = [n for n in names if "#" in n]
    if dup:
        print(f"NOTE: 同一症例疑い (fold分離未対応): {dup}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_metrics, all_t, all_p, all_prob = [], [], [], []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 5, num=10), cv=3, scoring="neg_log_loss",
            max_iter=2000, random_state=0, n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        prob_all = clf.predict_proba(X[te])
        pred = prob_all.argmax(axis=1)
        if len(labels) == 2:
            prob = prob_all[:, 1]
            roc_kwargs = {}
        else:
            prob = prob_all
            roc_kwargs = {"multi_class": "ovo", "average": "macro"}
        m = get_eval_metrics(
            y[te], pred, prob, roc_kwargs=roc_kwargs, prefix=f"fold{fi}",
        )
        fold_metrics.append({k.split("/")[-1]: float(v) for k, v in m.items()})
        all_t.append(y[te]); all_p.append(pred); all_prob.append(prob_all)
        print(f"fold{fi}: " + " ".join(f"{k}={v:.3f}" for k, v in fold_metrics[-1].items()))
    all_t = np.concatenate(all_t); all_p = np.concatenate(all_p); all_prob = np.concatenate(all_prob)
    pooled_prob = all_prob[:, 1] if len(labels) == 2 else all_prob
    pooled_roc = {} if len(labels) == 2 else {"multi_class": "ovo", "average": "macro"}
    pooled = get_eval_metrics(
        all_t, all_p, pooled_prob, roc_kwargs=pooled_roc, prefix="pooled",
    )
    pooled = {k.split("/")[-1]: float(v) for k, v in pooled.items()}
    keys = sorted(fold_metrics[0])
    summary = {
        "n": len(y), "dim": X.shape[1], "classes": labels,
        "fold_metrics": fold_metrics,
        "mean": {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys},
        "std": {k: float(np.std([f[k] for f in fold_metrics])) for k in keys},
        "pooled_oof": pooled,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("--- mean±std (5-fold) ---")
    for k in keys:
        print(f"{k:>12}: {summary['mean'][k]:.3f} ± {summary['std'][k]:.3f}")
    print(f"pooled OOF: " + " ".join(f"{k}={v:.3f}" for k, v in pooled.items()))
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
