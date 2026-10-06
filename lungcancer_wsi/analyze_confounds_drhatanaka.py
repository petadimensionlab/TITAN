#!/usr/bin/env python3
"""DrHatanaka 交絡検証:
  A) 症例IDでfold分離 (StratifiedGroupKFold) — 同一患者の術後/4期ペア漏洩を排除
  B) スキャン日(=染色/スキャンバッチ代理)でfold分離 — バッチ交絡の影響
  C) 同一患者の2検体が最近傍か (identity leakage 検定)
  D) 検体種別×バッチの分割表
入力: results/DrHatanaka_TITAN_m300/summary.csv + metadata.csv + *.pt
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).parent
D = ROOT / "results" / "DrHatanaka_TITAN_m300"
sys.path.insert(0, str(ROOT / "TITAN"))
from titan.utils import get_eval_metrics

LAB2ID = {"4期CRT後再発": 0, "術後再発": 1}


def load():
    meta = {r["filename"]: r for r in csv.DictReader(open(D / "metadata.csv", encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(D / "summary.csv", encoding="utf-8")) if r["status"] == "ok"]
    X, y, case, date, spec, names = [], [], [], [], [], []
    for r in rows:
        p = Path(r["pt_path"])
        if not p.exists():
            p = D / (Path(r["filename"]).stem.replace(" ", "_") + ".pt")
        d = torch.load(p, map_location="cpu", weights_only=True)
        X.append(np.asarray(d["embedding"] if isinstance(d, dict) else d, dtype=np.float64).ravel())
        y.append(LAB2ID[r["label"]])
        m = meta.get(r["filename"], {})
        case.append(m.get("case_id", ""))
        date.append(m.get("scan_date", ""))
        spec.append(r["label"])
        names.append(r["filename"])
    return Normalizer(norm="l2").fit_transform(np.stack(X)), np.array(y), np.array(case), np.array(date), spec, names


def cv_eval(X, y, groups, tag):
    from collections import Counter
    ng = len(set(groups))
    nsplit = min(5, ng)
    if nsplit < 2:
        print(f"[{tag}] groups={ng} -> CV不可")
        return None
    sgkf = StratifiedGroupKFold(n_splits=nsplit, shuffle=True, random_state=0)
    fold_metrics, all_t, all_p, all_prob = [], [], [], []
    degenerate = 0
    for fi, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(set(y[tr])) < 2:
            degenerate += 1
            print(f"[{tag}] fold{fi}: train単一クラス -> skip")
            continue
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 5, 10), cv=3, scoring="neg_log_loss",
                                   max_iter=2000, random_state=0, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])
        pred = prob.argmax(1)
        pr1 = prob[:, 1]
        m = get_eval_metrics(y[te], pred, pr1, prefix=f"f{fi}")
        fold_metrics.append({k.split("/")[-1]: float(v) for k, v in m.items()})
        all_t.append(y[te]); all_p.append(pred); all_prob.append(pr1)
        print(f"[{tag}] fold{fi}: n_tr={len(tr)} n_te={len(te)} test_classes={sorted(set(y[te]))} "
              + " ".join(f"{k}={v:.3f}" for k, v in fold_metrics[-1].items()))
    if not all_t:
        print(f"[{tag}] 有効foldなし (完全なバッチ分離のため)")
        return {"tag": tag, "degenerate_folds": degenerate, "valid_folds": 0}
    pooled = get_eval_metrics(np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_prob), prefix="p")
    pooled = {k.split("/")[-1]: float(v) for k, v in pooled.items()}
    keys = sorted(fold_metrics[0])
    res = {"tag": tag, "groups": ng, "valid_folds": len(fold_metrics), "degenerate_folds": degenerate,
           "mean": {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys},
           "std": {k: float(np.std([f[k] for f in fold_metrics])) for k in keys},
           "pooled": pooled}
    print(f"[{tag}] mean " + " ".join(f"{k}={res['mean'][k]:.3f}±{res['std'][k]:.3f}" for k in keys))
    return res


def main() -> int:
    X, y, case, date, spec, names = load()
    from collections import Counter
    print(f"n={len(y)} cases={len(set(case))} dates={len(set(date))}")
    print("label x date:")
    for d in sorted(set(date)):
        print(f"  {d}: {dict(Counter(spec[i] for i in range(len(y)) if date[i]==d))}")

    out = {}
    # A) 症例IDで分離
    out["group_by_case"] = cv_eval(X, y, case, "case")
    # B) スキャン日で分離
    out["group_by_scan_date"] = cv_eval(X, y, date, "scan_date")

    # C) 同一患者検定: 異ラベル最近傍が同一症例か
    sim = X @ X.T
    np.fill_diagonal(sim, -1)
    hit = 0; tot = 0
    for i in range(len(y)):
        mask = y != y[i]
        if mask.sum() == 0:
            continue
        j = np.where(mask)[0][np.argmax(sim[i][mask])]
        tot += 1
        if case[i] and case[i] == case[j]:
            hit += 1
    print(f"[同一患者検定] 異ラベル最近傍が同一症例: {hit}/{tot} = {hit/max(tot,1):.3f} (chance~{1/max(len(set(case))-1,1):.3f})")
    out["same_case_nn_rate"] = {"hit": hit, "total": tot, "rate": hit / max(tot, 1)}
    (D / "confound_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {D/'confound_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
