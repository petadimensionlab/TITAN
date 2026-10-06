#!/usr/bin/env python3
"""results/DrKitai_TITAN/summary.csv + *.pt からHTMLレポート生成 (torch不要でもCSV集計は可)。"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/DrKitai_TITAN/summary.csv"))
    ap.add_argument("--emb-dir", type=Path, default=Path("results/DrKitai_TITAN"))
    ap.add_argument("--out", type=Path, default=Path("results/DrKitai_TITAN/report.html"))
    args = ap.parse_args()
    if not args.input.exists():
        print(f"summaryがありません: {args.input}。先に analyze を実行してください。")
        return 2
    df = pd.read_csv(args.input)
    print(df["status"].value_counts().to_string())
    print(df.groupby("label")["status"].value_counts().to_string())
    # embedding scatter (PCA, sklearn任意)
    try:
        import numpy as np
        import torch
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        embs, labels, names = [], [], []
        for _, r in df[df.status=="ok"].iterrows():
            p = Path(r["pt_path"])
            if not p.exists():
                p = args.emb_dir / (Path(r["filename"]).stem.replace(" ", "_") + ".pt")
            if not p.exists(): continue
            d = torch.load(p, map_location="cpu")
            e = d["embedding"] if isinstance(d, dict) else d
            embs.append(np.asarray(e).ravel()); labels.append(r["label"]); names.append(r["filename"])
        if embs:
            X = np.stack(embs)
            xy = PCA(n_components=2).fit_transform(X)
            plt.figure(figsize=(7,6))
            for lab in sorted(set(labels)):
                import numpy as _np
                m = _np.array(labels)==lab
                plt.scatter(xy[m,0], xy[m,1], label=f"{lab} (n={m.sum()})", s=40, alpha=0.8)
            plt.legend(); plt.title(f"TITAN slide embeddings PCA (n={len(embs)}, dim={X.shape[1]})")
            plt.tight_layout()
            png = args.out.with_suffix(".pca.png")
            plt.savefig(png, dpi=150)
            print(f"PCA保存: {png}")
            # HTML
            rows = "\n".join(f"<tr><td>{n}</td><td>{l}</td></tr>" for n,l in zip(names, labels))
            args.out.write_text(f"<html><head><meta charset='utf-8'><title>DrKitai TITAN report</title></head><body><h1>DrKitai TITAN report (n={len(embs)})</h1><img src='{png.name}' style='max-width:800px'><h2>slides</h2><table border=1><tr><th>file</th><th>label</th></tr>{rows}</table><p>summary: {args.input}</p></body></html>", encoding="utf-8")
            print(f"HTML保存: {args.out}")
    except Exception as e:
        print(f"可視化スキップ: {e}")
        args.out.write_text(f"<html><body><h1>DrKitai TITAN report</h1><pre>{df.to_string()}</pre></body></html>", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
