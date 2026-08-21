# TITAN

> **A multimodal whole-slide foundation model for pathology** — *Nature Medicine*
>
> Original project: [mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN)

[日本語版 README はこちら / Japanese README](./README-jp.md)

---

## Overview

This repository is a **fork** of [MahmoodLab/TITAN](https://github.com/mahmoodlab/TITAN),
configured to run on **Apple Silicon (MPS) GPU** and managed with **uv**.
The original upstream README is preserved at [README-original.md](./README-original.md).

**TITAN** (Transformer-based pathology **I**mage and **T**ext **A**lignment **N**etwork)
is a multimodal whole-slide foundation model for computational pathology. It
encodes histopathology ROIs / whole slides into versatile feature
representations via self-supervised learning and vision-language alignment.

- **Paper**: [Nature Medicine (2025)](https://www.nature.com/articles/s41591-025-03982-3)
- **Model weights**: [MahmoodLab/TITAN on Hugging Face](https://huggingface.co/MahmoodLab/TITAN) (gated — request access)
- **Blog**: [Building vision-language guided multimodal whole-slide TITAN](https://www.linkedin.com/pulse/building-vision-language-guided-multimodal-whole-slide-tong-ding-0hawe/)

---

## Changes made in this fork

This fork differs from upstream in the following ways:

### 1. Environment management with `uv` (venv)

Upstream used `conda` + `pip install -e .`. This fork uses **uv**:

- `pyproject.toml` declares all runtime dependencies
- `uv.lock` pins the exact resolved dependency versions
- `.venv/` is created and managed by `uv`

### 2. Apple Silicon (MPS) GPU support

All CUDA-only code paths were made device-agnostic so PyTorch runs on the
**MPS (Metal) backend** on Apple Silicon automatically.

- `titan/utils.py`:
  - `get_device()` — returns `cuda` > `mps` > `cpu` in order of availability
  - `amp_dtype(device)` — `bfloat16` on CUDA, `float16` on MPS/CPU
- `titan/finetune.py`:
  - `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler(device.type, ...)`
  - `torch.cuda.amp.autocast` → `torch.autocast(device_type=device.type, ...)`
- `notebooks/*.ipynb` and `README.md` updated to use `get_device()` /
  `amp_dtype()` / device-agnostic `autocast`

### 3. Dependency fixes required for model loading

- **Pinned `transformers>=4.46,<5`** — TITAN's remote code
  (`modeling_titan.py`) is **incompatible with transformers 5.x**
  (fails with `AttributeError: 'Titan' object has no attribute 'all_tied_weights_keys'`).
  Verified working with `transformers 4.57.6` (model loads: 159M params).
- Added `[build-system]` (setuptools) so `uv sync` installs the `titan`
  package as editable — required for `python -m titan.finetune`.
- Removed an invalid empty `authors` email that broke the build.

---

## Installation

### Prerequisites

- macOS on **Apple Silicon** (M-series chip) — for MPS GPU support
- [uv](https://docs.astral.sh/uv/) installed (`brew install uv`)

### Setup

```bash
git clone <your-fork-url>/TITAN.git
cd TITAN

# create venv and install all dependencies from uv.lock
uv sync

# activate the environment
source .venv/bin/activate
```

> The model weights are **gated** on Hugging Face. Request access at
> https://huggingface.co/MahmoodLab/TITAN, then log in before first use:
>
> ```bash
> huggingface-cli login
> ```

---

## Device / GPU usage

Device selection is automatic — the same code runs on **CUDA**, **MPS** (Apple
Silicon GPU), or **CPU**:

```python
from titan.utils import get_device, amp_dtype

device = get_device()   # torch.device('mps') on Apple Silicon
print(device)           # mps
```

Mixed precision is device-agnostic via `amp_dtype(device)` (`bfloat16` on CUDA,
`float16` on MPS/CPU) and `torch.autocast(device_type=device.type, ...)`.

---

## Running inference

```python
from transformers import AutoModel
from titan.utils import amp_dtype, get_device

device = get_device()
model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)

# ... load patch features + coords (e.g. from a CLAM/CONCH extraction step) ...

with torch.autocast(device_type=device.type, dtype=amp_dtype(device)), torch.inference_mode():
    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
```

---

## Running finetune

> **Important**: run via `python -m titan.finetune`, **not** `python titan/finetune.py`
> (the latter fails to resolve the `titan` package import).

```bash
uv run python -m titan.finetune \
  --num_epochs 10 \
  --batch_size 1 \
  --num_workers 0 \
  --save_dir ./logs
```

The bundled `finetune.py` runs on a **dummy dataset** — replace the
`FinetuneDataset` class with your own data loader for real training.

---

## Known caveats (this fork)

| Issue | Workaround |
|---|---|
| `transformers` 5.x incompatible with TITAN remote code | `transformers` is pinned to `<5`; do not upgrade |
| `DataLoader(num_workers>0)` crashes on macOS | Use `--num_workers 0` |
| `num_epochs<=2` → `best_model_weights` is `None` TypeError at the end | Use `--num_epochs >= 3` (default is 10) |

### Measured performance on MPS (batch=1, fp16, fwd+backward)

| Patches / slide | ms / step |
|---|---|
| 36 | ~13 |
| 100 | ~15 |
| 400 | ~33 |
| 900 | ~80 |

Roughly **linear (~0.08 ms per patch)**. The bundled dummy finetune
(100 samples, 10 epochs, early-stop) completes in **~1 minute** on MPS.

---

## Demo notebooks

- `notebooks/inference_demo.ipynb` — slide embedding extraction
- `notebooks/zeroshot_demo.ipynb` — zero-shot classification
- `notebooks/linear_probe_demo.ipynb` — linear probing evaluation

---

## License and Terms of use

ⓒ Mahmood Lab. Released under the
[CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)
license — non-commercial, academic research use only with proper attribution.
Commercial use, sale, or monetization of the TITAN model and its derivatives is
prohibited. Downloading the model requires prior registration on Hugging Face
and agreement to the terms of use. See [README-original.md](./README-original.md)
for the full upstream terms.

## Reference

```
@article{ding2025multimodal,
  title={A multimodal whole-slide foundation model for pathology},
  author={Ding, Tong and Wagner, Sophia J and Song, Andrew H and Chen, Richard J and others},
  journal={Nature Medicine},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
