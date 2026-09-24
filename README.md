# CAMP-IE: minimal anonymous code

This anonymous repository contains only the core code needed to inspect the frozen multimodal conditioner and analyze attention heads. It intentionally omits benchmark evaluation, image generation experiments, plotting, judge models, and paper-table reproduction.

The implementation follows the paper's core mechanism:

1. load the Qwen-Image-Edit **Qwen2.5-VL conditioner** without the DiT/VAE;
2. hook each language self-attention head **before `o_proj` (`W_O`)** at the final conditioning token;
3. fit a per-head harmfulness direction from matched benign/harmful calibration pairs and benign-standardize the projection;
4. use matched `2 x 2` source/edit tuples to compute Edit, Source, and Compositional effects, corresponding to EHH, SHH, and CHH analysis.

The code is independently written. The public RHF implementation (Robust Harmful Features Under Jailbreak Attacks) was used only as a high-level reference for organizing a small mechanistic-analysis codebase around model loading, forward hooks, per-head activations, and analysis scripts; CAMP-IE uses a different multimodal model, activation definition, calibration construction, and factorial interaction analysis.

## Layout

```text
campie/
  modeling.py       Qwen-Image-Edit conditioner loading and input preparation
  hooks.py          discovery and pre-WO per-head activation hooks
  head_analysis.py  harmfulness projection and EHH/SHH/CHH factorial contrasts
scripts/
  inspect_model.py
  extract_conditioner_heads.py
  analyze_factorial_heads.py
data/
  manifest.example.csv
assets/
  qualitative assets supplied with the submission
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

A CUDA-capable environment is recommended for the Qwen2.5-VL conditioner. The default checkpoint is `Qwen/Qwen-Image-Edit`.

## 1. Inspect the conditioner

```bash
python scripts/inspect_model.py --model-id Qwen/Qwen-Image-Edit
```

The script reports the number of language layers, heads per layer, head dimension, and the resolved `self_attn.o_proj` modules. For the primary checkpoint used in the paper, the analysis is over the 28 x 28 language-attention heads.

## 2. Prepare a manifest

See `data/manifest.example.csv`. Two kinds of rows can coexist in one CSV:

- **direction calibration** rows use `direction_pair_id` and `direction_label` (`0=benign`, `1=harmful`), with exactly one matched pair per ID;
- **factorial analysis** rows use `tuple_id`, `source_level`, `edit_level`, and optionally `variant`.

For a factorial tuple, the four cells are:

```text
(S0, A0) neutral source + control edit
(S0, A1) neutral source + target edit
(S1, A0) critical source + control edit
(S1, A1) critical source + target edit
```

## 3. Extract conditioner head activations

```bash
python scripts/extract_conditioner_heads.py \
  --manifest data/my_manifest.csv \
  --image-root /path/to/images \
  --output outputs/activations.npz
```

No DiT denoising or image generation is run. The saved tensor is:

```text
activations: [request, layer, head, head_dim]
```

The hook records the concatenated attention output immediately before `o_proj`, reshaped back into individual heads.

## 4. Analyze EHH / SHH / CHH heads

```bash
python scripts/analyze_factorial_heads.py \
  --activations outputs/activations.npz \
  --output-dir outputs/head_analysis
```

For each head, the script computes

```text
E = 1/2 [(z01 - z00) + (z11 - z10)]
S = 1/2 [(z10 - z00) + (z11 - z01)]
C =      z11 - z10 - z01 + z00
```

where `z` is the benign-standardized projection onto the corresponding harmfulness direction. The script writes only analysis artifacts (`head_effects.npz` and `ranked_heads.json`); no evaluation metrics or figures are included.

## Assets

`assets/` contains the 68 JPEG qualitative assets supplied with the anonymous submission package. `assets/manifest.csv` records their paths and SHA-256 hashes. The code does not infer prompt text from these images.

## Scope

This package is deliberately minimal for anonymous review. It does **not** contain benchmark scoring, attack generation, adaptive evaluation, plotting, or paper-table scripts.
