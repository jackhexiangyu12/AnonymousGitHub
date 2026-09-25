# CAMP-IE: anonymous core implementation

This repository contains the compact pre-generation implementation used by CAMP-IE. It focuses on the model and representation-analysis path: loading the frozen multimodal conditioner, extracting pre-`W_O` attention-head features, matched factorial localization of EHH/SHH/CHH families, robust CHH selection, and fitting the Contrastive Interaction Subspace (CIS) with its scalar request-level readout.

Benchmark evaluation, image generation, plotting, paper-table reproduction, automated harmfulness judging, adaptive attack generation, and released-baseline wrappers are intentionally outside the scope of this compact anonymous core package.

## Method path

```text
source image + edit instruction
        |
frozen multimodal conditioner
        |
pre-W_O final-conditioning-token head outputs
        |
layer-specific harmfulness direction r_H^(ell)
        |
matched 2 x 2 E / S / C effects
        |
stable EHH / SHH / CHH localization
        |
robust CHH pool
        |
CHH concatenation -> benign whitening
        |
Sigma_E, Sigma_S, Sigma_C
        |
generalized eigensystem -> thin QR -> CIS
        |
L2 linear request-label readout
        |
development 5% benign-FPR threshold
        |
ALLOW / BLOCK before DiT denoising
```

The implementation is independently written. RHF is useful as a general reference for attention-head activation analysis, but CAMP-IE uses a multimodal source-edit factorial objective rather than RHF's head taxonomy or scoring rule.

## What is fixed explicitly

`campie/protocol.py` records the numerical protocol used by the core implementation:

- head output is recorded immediately before `W_O` at the fixed final conditioning token;
- harmfulness direction is **one direction per conditioner layer**, estimated as a paired harmful-minus-benign mean difference after within-layer head averaging;
- benign mean and standard deviation remain head-specific;
- localization uses matched source-group bootstrap resampling (`R_boot=10,000` by default);
- CHH deployment selection uses `Freq_h >= 0.80`, `SignCons_h >= 0.80`, and `CS_h <= 0.62`;
- paired bootstrap confidence intervals are reported descriptively and are **not** multiplicity-corrected or used as the deployment selection rule by default;
- whitening uses `epsilon_B = 0.05 tr(Sigma_B) / d_CIS`;
- the main-effect penalty uses `gamma = 0.10 tr(Sigma_E + Sigma_S) / d_CIS`;
- CIS rank is chosen from the training generalized eigenspectrum using the profile-likelihood scree criterion when `--rank auto` is used;
- the CHH fraction is selected from `{0.125, 0.25, 0.5, 0.75, 1.0}` on development request labels when `--rho auto` is used;
- logistic readout `C` is selected from `{0.1, 1, 10}` on development request-label log loss when `--readout-c auto` is used;
- the block threshold is calibrated separately on benign development scores for empirical FPR at or below 5%.

The fitted state and its companion JSON report record the number of whitening vectors, the number and rank of E/S/C contrast vectors before regularization, `epsilon_B`, `gamma`, selected rank, selected CHH fraction, selected readout regularization, and signed CIS-coordinate summaries. These are method diagnostics rather than benchmark metrics.

## Repository layout

```text
campie/
  modeling.py       Qwen-Image-Edit conditioner loading via the official Diffusers input path
  hooks.py          pre-W_O recording; additive residual and natural-donor replacement hooks
  data.py           factorial-manifest and train/dev/test separation checks
  protocol.py       fixed numerical protocol
  head_analysis.py  layer directions, E/S/C effects, bootstrap stability, carrier robustness
  cis.py            CHH concatenation, whitening, CIS, readout, threshold, fit diagnostics
scripts/
  inspect_model.py
  extract_conditioner_heads.py
  analyze_factorial_heads.py
  fit_cis.py
  score_requests.py
  screen_request.py
data/
  manifest.example.csv
assets/
  qualitative assets included with the anonymous submission
tests/
  test_core.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

A CUDA-capable environment is recommended for conditioner extraction.

## 1. Inspect the conditioner

```bash
python scripts/inspect_model.py \
  --model-id Qwen/Qwen-Image-Edit \
  --revision YOUR_EXACT_CHECKPOINT_REVISION
```

The extractor resolves the language-model `self_attn.o_proj` modules and records their input, i.e. concatenated per-head output before `W_O`.

### Official Qwen input path

CAMP-IE does not maintain a second copy of the Qwen-Image-Edit prompt wrapper or image-resizing policy. `campie/modeling.py` constructs a conditioning-only `diffusers.QwenImageEditPipeline` from the checkpoint text encoder, tokenizer, and processor, enters the pipeline's normal `__call__` path, and terminates immediately after the pipeline's own `encode_prompt` returns. Consequently, text serialization and source-image preprocessing are inherited from the installed official Diffusers implementation rather than redefined in this repository. The activation archive records `conditioning_backend`, `diffusers_version`, and `transformers_version` together with the model checkpoint revision.

## 2. Prepare the manifest

See `data/manifest.example.csv`. Relevant fields are:

- `direction_pair_id`, `direction_label`: pair-aligned harmful/benign direction-calibration examples;
- `tuple_id`, `source_level`, `edit_level`, `variant`: matched factorial groups;
- `risk_label`: request-level label, with only `(S1,A1)` positive inside a factorial tuple;
- `split`, `source_group`, `normalized_edit_template`: train/dev/test separation metadata.

Each `(tuple_id, variant)` contains exactly four cells:

```text
(S0,A0) neutral source + control edit
(S0,A1) neutral source + target edit
(S1,A0) critical source + control edit
(S1,A1) critical source + target edit
```

`validate_manifest` rejects incomplete 2x2 groups, duplicated factorial cells, malformed calibration pairs, source-group leakage across train/dev/test, and normalized-template leakage across train/dev/test.

## 3. Extract pre-W_O head activations

```bash
python scripts/extract_conditioner_heads.py \
  --manifest data/my_manifest.csv \
  --image-root /path/to/images \
  --output outputs/activations.npz \
  --revision YOUR_EXACT_CHECKPOINT_REVISION
```

The main activation tensor has shape `[request, layer, head, head_dim]`. The saved file also records the exact model ID/revision, model head geometry, manifest row count, calibration-pair count, and factorial tuple counts.

## 4. Localize EHH / SHH / CHH families

```bash
python scripts/analyze_factorial_heads.py \
  --activations outputs/activations.npz \
  --output-dir outputs/head_analysis \
  --bootstrap 10000
```

For every matched tuple/variant:

```text
E = 1/2 [(z01-z00) + (z11-z10)]
S = 1/2 [(z10-z00) + (z11-z01)]
C =      z11-z10-z01+z00
```

`head_selection.json` records, for each retained head, the mean standardized effect, paired bootstrap interval, source-group recurrence, carrier sign consistency, and carrier sensitivity. Carrier comparisons are matched to the canonical realization on the same tuple set, so missing carrier variants cannot change the reference population silently.

The default deployment rule does not use the bootstrap interval as an additional significance filter. Use `--require-positive-ci` only for a stricter sensitivity analysis.

## 5. Fit CIS

```bash
python scripts/fit_cis.py \
  --activations outputs/activations.npz \
  --head-selection outputs/head_analysis/head_selection.json \
  --output outputs/cis_state.npz \
  --rho auto \
  --rank auto \
  --readout-c auto
```

The fit is performed only from training activations/request labels plus development labels used for hyperparameter selection and threshold calibration. No generated-output harmfulness score or native-guard decision is used by the readout.

Alongside `cis_state.npz`, the script writes `cis_state.json`. The JSON includes:

- number of selected CHHs and resulting `d_CIS`;
- number of benign whitening vectors and empirical covariance rank before regularization;
- number of E/S/C contrast vectors and their row ranks;
- numerical `epsilon_B` and `gamma`;
- selected generalized-eigenspectrum rank;
- development-selected `rho` and logistic `C`;
- signed mean/std of each CIS coordinate for the four factorial cells and for the interaction contrast.

The signed coordinate report is included to distinguish the unsupervised interaction-energy subspace from the supervised request-label readout; it is not an evaluation metric.

## 6. One-pass pre-generation decision

For previously extracted activations:

```bash
python scripts/score_requests.py \
  --activations outputs/requests.npz \
  --state outputs/cis_state.npz \
  --output outputs/request_scores.csv
```

For a single source/edit request:

```bash
python scripts/screen_request.py \
  --image path/to/source.jpg \
  --instruction "edit instruction" \
  --state outputs/cis_state.npz \
  --revision YOUR_EXACT_CHECKPOINT_REVISION
```

`screen_request.py` performs one normal conditioning pass, reads the selected pre-`W_O` heads, and returns `ALLOW` or `BLOCK`. It deliberately does not invoke the DiT. In the complete image-editing service, an `ALLOW` decision forwards the original conditioning state unchanged; `BLOCK` terminates before denoising.

## Intervention primitive

`HeadOutputPatcher` supports two operations at the same fixed pre-`W_O` token:

- `mode="add"` with negative `alpha` for interaction-residual subtraction;
- `mode="replace"` for natural-donor replacement with an observed head vector.

The repository contains only the activation-editing primitive, not the generation/evaluation experiment around it.

## Scope and assets

`assets/` contains the qualitative examples supplied with the anonymous submission. Sensitive attack-generation templates, adaptive-search code, uncensored harmful outputs, benchmark judging pipelines, and external guard wrappers are not included in this compact anonymous repository.
