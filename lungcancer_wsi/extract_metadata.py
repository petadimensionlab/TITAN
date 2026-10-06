#!/usr/bin/env python3
"""WSIメタデータ抽出: 症例ID・検体種別・スキャン日/バッチ・NDPI属性。

症例IDは任意の対応表 (xlsx) から、2列を1症例としてペアリングして作成できる。
検体種別ラベルは --label-map (prefix->label のJSON) から付与する。

使い方:
  python extract_metadata.py --input data/<cohort> --out results/<cohort>/metadata.csv
  # 症例対応表を使う場合 (A列・D列を同一症例として扱う):
  python extract_metadata.py --input data/<cohort> --xlsx cases.xlsx \
      --xlsx-sheet 1 --col-a 0 --col-d 3 --label-map labels.json \
      --out results/<cohort>/metadata.csv
出力列: filename, stem, case_id, specimen, batch, scan_date, scan_hhmm,
        objective, mpp_x, macro_barcode
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from analyze_wsi_titan import find_slides, load_label_map


def build_case_map(xlsx: Path, sheet, col_a: int, col_d: int) -> dict[str, int]:
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[sheet] if isinstance(sheet, int) \
        else openpyxl.load_workbook(xlsx, data_only=True)[sheet]
    m: dict[str, int] = {}
    for i, row in enumerate(ws.iter_rows(min_row=3, values_only=True), start=1):
        for c in (col_a, col_d):
            v = row[c] if c < len(row) else None
            if v and isinstance(v, str) and v.strip():
                m[v.strip()] = i
    return m


def match_case(name: str, case_map: dict[str, int]) -> int | None:
    stem = Path(name).stem
    ks = [k for k in case_map if stem.startswith(k)]
    return case_map[max(ks, key=len)] if ks else None


def main() -> int:
    ap = argparse.ArgumentParser(description="WSIメタデータ抽出")
    ap.add_argument("--input", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "wsi_titan" / "metadata.csv")
    ap.add_argument("--xlsx", type=Path, default=None, help="症例対応表 (任意)")
    ap.add_argument("--xlsx-sheet", default=0, help="シート名 or インデックス (default: 0)")
    ap.add_argument("--col-a", type=int, default=0, help="症例A列 (0始まり)")
    ap.add_argument("--col-d", type=int, default=3, help="症例D列 (0始まり)")
    ap.add_argument("--label-map", type=Path, default=None, help="prefix->label のJSON")
    ap.add_argument("--batch-regex", default=r"^([A-Z]{2}\d{2})", help="ファイル名からバッチを抜く正規表現")
    args = ap.parse_args()

    sheet = args.xlsx_sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    case_map = build_case_map(args.xlsx, sheet, args.col_a, args.col_d) if args.xlsx else {}
    label_map = load_label_map(args.label_map)

    try:
        import openslide
    except ImportError:
        openslide = None

    def label_of(name: str) -> str:
        stem = Path(name).stem
        ks = [k for k in label_map if stem.startswith(k)]
        return label_map[max(ks, key=len)] if ks else "unknown"

    rows = []
    for p in find_slides(args.input):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})\.(\d{2})", p.name)
        scan = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
        hh = m.group(4) if m else ""
        bm = re.match(args.batch_regex, p.stem)
        batch = bm.group(1) if bm else ""
        barcode = obj = mpp = ""
        if openslide:
            try:
                s = openslide.OpenSlide(str(p))
                pr = s.properties
                barcode = pr.get("hamamatsu.Exposure.Barcode.Macro", "") or pr.get("openslide.barcode", "")
                obj = pr.get("openslide.objective-power", "")
                mpp = pr.get("openslide.mpp-x", "")
                s.close()
            except Exception as e:
                barcode = f"ERR:{type(e).__name__}"
        rows.append({
            "filename": p.name, "stem": p.stem,
            "case_id": match_case(p.name, case_map) or "",
            "specimen": label_of(p.name), "batch": batch,
            "scan_date": scan, "scan_hhmm": hh,
            "objective": obj, "mpp_x": mpp, "macro_barcode": barcode,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"saved {args.out} ({len(rows)} rows)")
    print("specimen:", dict(Counter(r["specimen"] for r in rows)))
    print("case_id distinct:", len(set(r["case_id"] for r in rows if r["case_id"])) or "(なし)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
