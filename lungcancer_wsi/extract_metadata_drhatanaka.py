#!/usr/bin/env python3
"""DrHatanaka メタデータ抽出: 症例ID(行)・検体種別・スキャン日/バッチ・NDPI属性。
出力: results/DrHatanaka_TITAN_m300/metadata.csv
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from analyze_drkitai_titan import find_slides

XLSX = ROOT / "data" / "DrHatanaka" / "術後再発_4期CRT後再発WSIリスト.xlsx"
OUT = ROOT / "results" / "DrHatanaka_TITAN_m300" / "metadata.csv"


def build_case_map() -> dict[str, int]:
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, data_only=True)["番号あり"]
    m: dict[str, int] = {}
    for i, row in enumerate(ws.iter_rows(min_row=3, values_only=True), start=1):
        a, d = row[0], row[3]
        if a and isinstance(a, str):
            m[a.strip()] = i
        if d and isinstance(d, str):
            m[d.strip()] = i
    return m


def match(name: str, case_map: dict[str, int]) -> int | None:
    stem = Path(name).stem
    ks = [k for k in case_map if stem.startswith(k)]
    return case_map[max(ks, key=len)] if ks else None


def main() -> int:
    case_map = build_case_map()
    try:
        import openslide
    except ImportError:
        openslide = None

    rows = []
    for p in find_slides(ROOT / "data" / "DrHatanaka"):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})\.(\d{2})", p.name)
        scan = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
        hh = m.group(4) if m else ""
        spec = "切除(術後)" if re.match(r"^\d+[-P]", p.stem) else "生検(4期CRT)"
        dxbatch = ""
        bm = re.match(r"^(DX\d{2})\d*", p.stem)
        if bm:
            dxbatch = bm.group(1)
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
            "case_id": match(p.name, case_map) or "",
            "specimen": spec, "dx_batch": dxbatch,
            "scan_date": scan, "scan_hhmm": hh,
            "objective": obj, "mpp_x": mpp, "macro_barcode": barcode,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT} ({len(rows)} rows)")
    from collections import Counter
    print("specimen:", Counter(r["specimen"] for r in rows))
    print("case num:", len(set(r["case_id"] for r in rows if r["case_id"])), "distinct")
    print("scan_date->specimen:")
    for d in sorted(set(r["scan_date"] for r in rows)):
        c = Counter(r["specimen"] for r in rows if r["scan_date"] == d)
        print(f"  {d}: {dict(c)}")
    print("objective:", Counter(r["objective"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
