# CAMP-IE

Anonymous implementation for **CAMP-IE: Understanding Safety Risks in Multimodal Instruction Following Image Editing Models**.

This repository contains the core code used to extract attention-head features from the Qwen-Image-Edit conditioner, identify EHH/SHH/CHH head families with the matched factorial design, fit the Contrastive Interaction Subspace (CIS), and score requests before image generation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

A CUDA-capable environment is recommended for feature extraction.

## Repository

```text
campie/
  modeling.py       model loading and conditioning
  hooks.py          pre-W_O attention-head hooks
  data.py           manifest checks and split validation
  head_analysis.py  EHH / SHH / CHH analysis
  cis.py            CIS fitting and request scoring

scripts/
  extract_conditioner_heads.py
  analyze_factorial_heads.py
  fit_cis.py
  score_requests.py
  screen_request.py

assets/             qualitative examples
```

## Data

`data/manifest.example.csv` shows the expected manifest format. Each factorial group contains the four matched cells `(S0,A0)`, `(S0,A1)`, `(S1,A0)`, and `(S1,A1)`. Source groups and normalized edit templates are kept disjoint across train, development, and test splits.

## Feature extraction

```bash
python scripts/extract_conditioner_heads.py \
  --manifest data/my_manifest.csv \
  --image-root /path/to/images \
  --output outputs/activations.npz \
  --revision YOUR_EXACT_CHECKPOINT_REVISION
```

The extractor uses the official Diffusers Qwen-Image-Edit conditioning path and records the input to each attention `o_proj`, i.e. the per-head representation before `W_O`.

## Head analysis

```bash
python scripts/analyze_factorial_heads.py \
  --activations outputs/activations.npz \
  --output-dir outputs/head_analysis
```

For each matched group, the edit, source, and interaction effects are

```text
E = 1/2 [(z01-z00) + (z11-z10)]
S = 1/2 [(z10-z00) + (z11-z01)]
C =      z11-z10-z01+z00
```

The analysis script also applies the recurrence and carrier-robustness criteria used for CHH selection.

## Fit CIS

```bash
python scripts/fit_cis.py \
  --activations outputs/activations.npz \
  --head-selection outputs/head_analysis/head_selection.json \
  --output outputs/cis_state.npz \
  --rho auto \
  --rank auto \
  --readout-c auto
```

This fits benign whitening, the generalized interaction subspace, and the linear request-level readout. The decision threshold is calibrated on the development split.

## Screen a request

```bash
python scripts/screen_request.py \
  --image path/to/source.jpg \
  --instruction "edit instruction" \
  --state outputs/cis_state.npz \
  --revision YOUR_EXACT_CHECKPOINT_REVISION
```

The script returns `ALLOW` or `BLOCK` from one conditioning pass and does not run the DiT generator.

## Notes

The repository is intentionally limited to the core CAMP-IE implementation. Benchmark evaluation, plotting, automated judging, adaptive attack generation, and external baseline wrappers are not included. The `assets/` directory contains the qualitative examples used in the anonymous submission.
