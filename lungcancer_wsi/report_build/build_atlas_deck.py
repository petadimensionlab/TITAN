#!/usr/bin/env python3
"""atlas_manifest.json からアトラス版 Marp デッキ (03_atlas.md) を生成する.

使い方: python3 lungcancer_wsi/report_build/build_atlas_deck.py
入力: viewer/atlas/atlas_manifest.json (make_atlas.py の出力)
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent           # report_build/
ATLAS_DIR = HERE.parent / "viewer" / "atlas"
MANIFEST = ATLAS_DIR / "atlas_manifest.json"
OUT = HERE / "markdown" / "03_atlas.md"

CLS_JA = {"LUAD": "肺腺癌", "LUSC": "扁平上皮癌", "LYMPH": "リンパ球/炎症細胞",
          "STROMA": "線維間質", "ALV": "肺胞腔/背景", "NECR": "壊死", "BLD": "血液"}
ORDER = ["LUAD", "LUSC", "LYMPH", "STROMA", "ALV", "NECR", "BLD"]

HEADER = """---
marp: true
theme: default
paginate: true
lang: ja
style: |
  section { font-family: "Noto Sans CJK JP", sans-serif; font-size: 26px;
            background: white; color: #111; padding: 28px 60px; }
  h1 { color: #c8102e; font-size: 38px; margin-bottom: 6px; }
  img.patch { height: 330px; width: 330px; object-fit: cover;
              border: 1px solid #ccc; }
  img.ov { height: 125px; max-width: 350px; border: 1px solid #ccc; }
  .row { display: flex; gap: 40px; align-items: flex-start;
         justify-content: center; }
  .cell { text-align: center; }
  .cap { font-size: 17px; color: #444; margin-top: 4px; }
  ul { font-size: 24px; }
---

# 画像アトラス — 組織分類を画像で見る

## CONCH zero-shot が選んだクラス別代表パッチ

**患者さんの画像を含むため、配布範囲は限定された資料です。**

---

# 読み方

- 各画像は 512×512 ピクセルのパッチ（約 230 マイクロメートル四方、20倍物鏡相当）
- 左下の黒いバーは 100 マイクロメートルのスケールバー
- 「位置」図は同じスライドの低倍率全景で、赤枠がパッチの場所
- パッチは各クラスで **CONCH の分類確率が最も高い例** を採用
- **CONCH の判定は正解診断ではなく、見た目の類似度による候補**です

---
"""


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def class_slide(cid: str, items: list[dict]) -> str:
    ja = CLS_JA[cid]
    parts = [f"# {ja}（{cid}）\n", '<div class="row">']
    for it in items:
        parts.append(
            f'<div class="cell">\n'
            f'<img class="patch" src="{data_uri(ATLAS_DIR / it["patch"])}"/>\n'
            f'<div class="cap">例: {it["stem"]} ・ CONCH確率 {it["prob"]:.2f}</div>\n'
            f'<img class="ov" src="{data_uri(ATLAS_DIR / f"overview_{it["stem"]}.png")}"/>\n'
            f'<div class="cap">位置（赤枠）: {it["stem"]}</div>\n'
            f"</div>"
        )
    parts.append("</div>\n\n---\n")
    return "\n".join(parts)


def summary_slide(items: list[dict]) -> str:
    stems = sorted({it["stem"] for it in items})
    n = len(items)
    return f"""# まとめと注意

- 7クラス × 最大2例、計 {n} パッチ（全数グリッド {len(stems)} スライドから選択）
- 各例はクラス内で分類確率上位のため、「典型例」であって「難所」は示していません
- 確率が高い例でも誤りの可能性はあります（正解診断との照合は未実施＝**未確認**）
- 境界域・混在領域の例は次回以降の検証対象です

---
"""


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_cls = {}
    for it in manifest:
        by_cls.setdefault(it["class"], []).append(it)

    slides = [HEADER]
    for cid in ORDER:
        if cid in by_cls:
            slides.append(class_slide(cid, by_cls[cid]))
    slides.append(summary_slide(manifest))

    OUT.write_text("\n".join(slides).rstrip().removesuffix("---").rstrip() + "\n",
                   encoding="utf-8")
    print(f"-> {OUT}: {1 + 2 + sum(len(v) for v in by_cls.values()) // 1 + 1} sections")


if __name__ == "__main__":
    main()
