#!/usr/bin/env python3
"""埋め込みの交絡検証:
  A) 症例IDでfold分離 (StratifiedGroupKFold) — 同一患者の複数検体の漏洩を排除
  B) スキャン日/バッチでfold分離 — バッチ交絡の影響
  C) 同一患者の複数検体が最近傍か (identity leakage 検定)
  D) ラベル × グループの分割表
入力: <emb-dir>/summary.csv + <metadata> + *.pt
出力: <emb-dir>/confound_analysis.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "TITAN"))
from titan.utils import get_eval_metrics


def cv_eval(X, y, groups, tag):
    ng = len(set(groups))
    nsplit = min(5, ng)
    if nsplit < 2:
        print(f"[{tag}] groups={ng} -> CV不可")
        return None
    sgkf = StratifiedGroupKFold(n_splits=nsplit, shuffle=True, random_state=0)
    folds, all_t, all_p, all_prob = [], [], [], []
    degen = 0
    for fi, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(set(y[tr])) < 2:
            degen += 1
            print(f"[{tag}] fold{fi}: train単一クラス -> skip")
            continue
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 5, 10), cv=3, scoring="neg_log_loss",
                                   max_iter=2000, random_state=0, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])
        pred = prob.argmax(1)
        pr1 = prob[:, 1] if prob.shape[1] == 2 else prob
        m = get_eval_metrics(y[te], pred, pr1, prefix=f"f{fi}")
        folds.append({k.split("/")[-1]: float(v) for k, v in m.items()})
        all_t.append(y[te]); all_p.append(pred); all_prob.append(prob)
        print(f"[{tag}] fold{fi}: n_tr={len(tr)} n_te={len(te)} "
              + " ".join(f"{k}={v:.3f}" for k, v in folds[-1].items()))
    if not all_t:
        print(f"[{tag}] 有効foldなし")
        return {"tag": tag, "degenerate_folds": degen, "valid_folds": 0}
    pooled = get_eval_metrics(np.concatenate(all_t), np.concatenate(all_p), prefix="p")
    pooled = {k.split("/")[-1]: float(v) for k, v in pooled.items()}
    keys = sorted(folds[0])
    res = {"tag": tag, "groups": ng, "valid_folds": len(folds), "degenerate_folds": degen,
           "mean": {k: float(np.mean([f[k] for f in folds])) for k in keys},
           "std": {k: float(np.std([f[k] for f in folds])) for k in keys}, "pooled": pooled}
    print(f"[{tag}] mean " + " ".join(f"{k}={res['mean'][k]:.3f}±{res['std'][k]:.3f}" for k in keys))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="埋め込みの交絡検証")
    ap.add_argument("--emb-dir", type=Path, default=ROOT / "results" / "wsi_titan")
    ap.add_argument("--metadata", type=Path, default=None, help="metadata.csv (default: <emb-dir>/metadata.csv)")
    ap.add_argument("--case-col", default="case_id")
    ap.add_argument("--date-col", default="scan_date")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    D = args.emb_dir
    meta_path = args.metadata or D / "metadata.csv"
    OUT = args.out or D / "confound_analysis.json"

    meta = {r["filename"]: r for r in csv.DictReader(open(meta_path, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(D / "summary.csv", encoding="utf-8")) if r["status"] == "ok"]
    labels = sorted(set(r["label"] for r in rows))
    lab2id = {l: i for i, l in enumerate(labels)}
    X, y, case, date, spec = [], [], [], [], []
    for r in rows:
        p = Path(r["pt_path"])
        if not p.exists():
            p = D / (Path(r["filename"]).stem.replace(" ", "_") + ".pt")
        d = torch.load(p, map_location="cpu", weights_only=True)
        X.append(np.asarray(d["embedding"] if isinstance(d, dict) else d, dtype=np.float64).ravel())
        y.append(lab2id[r["label"]])
        m = meta.get(r["filename"], {})
        case.append(m.get(args.case_col, ""))
        date.append(m.get(args.date_col, ""))
        spec.append(r["label"])
    X = Normalizer(norm="l2").fit_transform(np.stack(X))
    y = np.array(y)
    print(f"n={len(y)} labels={labels} cases={len(set(case))} dates={len(set(date))}")

    out = {"group_by_case": cv_eval(X, y, np.array(case), "case"),
           "group_by_date": cv_eval(X, y, np.array(date), "scan_date")}

    sim = X @ X.T
    np.fill_diagonal(sim, -1)
    hit = tot = 0
    for i in range(len(y)):
        mask = y != y[i]
        if mask.sum() == 0:
            continue
        j = np.where(mask)[0][np.argmax(sim[i][mask])]
        tot += 1
        if case[i] and case[i] == case[j]:
            hit += 1
    print(f"[同一患者検定] 異ラベル最近傍が同一症例: {hit}/{tot} = {hit/max(tot,1):.3f}")
    out["same_case_nn_rate"] = {"hit": hit, "total": tot, "rate": hit / max(tot, 1)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
