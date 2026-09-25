from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .protocol import DEFAULT_PROTOCOL


@dataclass
class HeadDirection:
    # One direction per conditioner layer, shared by heads in that layer.
    direction: np.ndarray   # [L,Dh]
    benign_mean: np.ndarray # [L,H]
    benign_std: np.ndarray  # [L,H]
    eps: float


@dataclass
class FactorialHeadEffects:
    edit: np.ndarray        # [G,L,H]
    source: np.ndarray      # [G,L,H]
    interaction: np.ndarray # [G,L,H]
    groups: list[tuple[str, str]]


@dataclass(frozen=True)
class StableHead:
    layer: int
    head: int
    mean_effect: float
    ci_low: float
    ci_high: float
    recurrence: float
    sign_consistency: float
    carrier_sensitivity: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def fit_layer_harmfulness_direction(
    harmful: np.ndarray,
    benign: np.ndarray,
    eps_scale: float = DEFAULT_PROTOCOL.eps_z_scale,
) -> HeadDirection:
    """Fit one harmfulness direction r_H^(ell) per conditioner layer.

    ``harmful`` and ``benign`` are pair-aligned tensors with shape
    ``[N, layer, head, head_dim]``. The estimator first averages the head vectors
    within a layer for each matched request and then takes the mean paired
    harmful-minus-benign difference across calibration pairs. This makes the
    estimator explicitly layer-specific rather than head-specific.

    Every head in a layer is projected onto the same unit direction. Benign mean
    and standard deviation are still estimated separately for each head.
    """
    h = np.asarray(harmful, dtype=np.float64)
    b = np.asarray(benign, dtype=np.float64)
    if h.shape != b.shape or h.ndim != 4:
        raise ValueError("harmful and benign must have identical shape [N,L,H,Dh]")
    if h.shape[0] < 2:
        raise ValueError("at least two matched calibration pairs are required")

    h_layer = h.mean(axis=2)  # [N,L,Dh]
    b_layer = b.mean(axis=2)  # [N,L,Dh]
    direction = np.mean(h_layer - b_layer, axis=0)  # [L,Dh]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-12)

    benign_raw = np.einsum("nlhd,ld->nlh", b, direction)
    mu = benign_raw.mean(axis=0)
    sigma = benign_raw.std(axis=0, ddof=1)
    positive = sigma[sigma > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = float(eps_scale * scale)
    return HeadDirection(direction=direction, benign_mean=mu, benign_std=sigma, eps=eps)


def project_head_scores(activations: np.ndarray, state: HeadDirection) -> np.ndarray:
    """Project head outputs onto layer directions and benign-standardize."""
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("activations must have shape [N,L,H,Dh]")
    if x.shape[1] != state.direction.shape[0] or x.shape[-1] != state.direction.shape[1]:
        raise ValueError("activation shape is incompatible with harmfulness directions")
    if x.shape[1:3] != state.benign_mean.shape:
        raise ValueError("activation shape is incompatible with benign statistics")
    raw = np.einsum("nlhd,ld->nlh", x, state.direction)
    return (raw - state.benign_mean[None]) / (state.benign_std[None] + state.eps)


def factorial_head_effects(
    scores: np.ndarray,
    tuple_id: Iterable,
    source_level: Iterable,
    edit_level: Iterable,
    variant: Iterable | None = None,
) -> FactorialHeadEffects:
    """Compute edit, source, and source-edit interaction effects.

    For every matched tuple/variant:
      E = 1/2[(z01-z00) + (z11-z10)]
      S = 1/2[(z10-z00) + (z11-z01)]
      C =      z11-z10-z01+z00
    """
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError("scores must have shape [N,L,H]")

    tids = np.asarray(list(tuple_id)).astype(str)
    s = np.asarray(list(source_level), dtype=int)
    a = np.asarray(list(edit_level), dtype=int)
    v = np.asarray(["canonical"] * len(tids) if variant is None else list(variant)).astype(str)
    if not (len(tids) == len(s) == len(a) == len(v) == z.shape[0]):
        raise ValueError("metadata length mismatch")

    edits, sources, interactions, groups = [], [], [], []
    for tid in sorted(set(tids)):
        variants = sorted(set(v[tids == tid]))
        for vv in variants:
            mask = (tids == tid) & (v == vv)
            cells: dict[tuple[int, int], np.ndarray] = {}
            for sv in (0, 1):
                for av in (0, 1):
                    cell = mask & (s == sv) & (a == av)
                    if not np.any(cell):
                        raise ValueError(f"missing factorial cell ({sv},{av}) for tuple={tid}, variant={vv}")
                    cells[(sv, av)] = z[cell].mean(axis=0)

            z00, z01 = cells[(0, 0)], cells[(0, 1)]
            z10, z11 = cells[(1, 0)], cells[(1, 1)]
            edits.append(0.5 * ((z01 - z00) + (z11 - z10)))
            sources.append(0.5 * ((z10 - z00) + (z11 - z01)))
            interactions.append(z11 - z10 - z01 + z00)
            groups.append((str(tid), str(vv)))

    if not groups:
        raise ValueError("no complete factorial tuple/variant groups were found")
    return FactorialHeadEffects(
        edit=np.stack(edits),
        source=np.stack(sources),
        interaction=np.stack(interactions),
        groups=groups,
    )


def _effect_eps(effect: np.ndarray, scale: float = DEFAULT_PROTOCOL.eps_effect_scale) -> float:
    std = np.std(effect, axis=0, ddof=1)
    positive = std[std > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    return float(scale * median)


def _group_effect_by_tuple(effect: np.ndarray, groups: list[tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    tids = np.asarray([g[0] for g in groups], dtype=str)
    unique = sorted(set(tids))
    per_tuple = []
    for tid in unique:
        per_tuple.append(effect[tids == tid].mean(axis=0))
    return np.stack(per_tuple), unique


def bootstrap_head_statistics(
    effect: np.ndarray,
    groups: list[tuple[str, str]],
    n_resamples: int = DEFAULT_PROTOCOL.development_bootstrap_resamples,
    seed: int = 0,
    ci_level: float = DEFAULT_PROTOCOL.bootstrap_ci_level,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return recurrence and paired bootstrap percentile intervals per head.

    Resampling is over matched tuples/source groups. The interval is descriptive;
    deployment selection is controlled by recurrence and carrier-stability rules.
    No multiplicity correction is applied by default.
    """
    x, _ = _group_effect_by_tuple(np.asarray(effect, dtype=np.float64), groups)
    n = x.shape[0]
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must lie in (0,1)")
    if n < 2:
        mean = x.mean(axis=0)
        recurrence = (mean > 0).astype(np.float64)
        return recurrence, mean.copy(), mean.copy()

    rng = np.random.default_rng(seed)
    draws = np.empty((n_resamples,) + x.shape[1:], dtype=np.float32)
    done = 0
    while done < n_resamples:
        r = min(chunk_size, n_resamples - done)
        idx = rng.integers(0, n, size=(r, n))
        draws[done:done + r] = x[idx].mean(axis=1).astype(np.float32)
        done += r

    recurrence = np.mean(draws > 0, axis=0)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha], axis=0)
    return recurrence.astype(np.float64), low.astype(np.float64), high.astype(np.float64)


def bootstrap_recurrence(
    effect: np.ndarray,
    groups: list[tuple[str, str]],
    n_resamples: int = DEFAULT_PROTOCOL.development_bootstrap_resamples,
    seed: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    recurrence, _, _ = bootstrap_head_statistics(
        effect, groups, n_resamples=n_resamples, seed=seed, chunk_size=chunk_size
    )
    return recurrence


def carrier_robustness(
    effect: np.ndarray,
    groups: list[tuple[str, str]],
    canonical_name: str = "canonical",
) -> tuple[np.ndarray, np.ndarray]:
    """Return SignCons_h and CS_h from matched carrier variants.

    Each non-canonical variant is compared against the canonical realization on
    the *same set of tuples*. Sign consistency is the fraction of variant types
    whose mean effect preserves the matched canonical sign. Carrier sensitivity
    is the largest matched variant change standardized by the canonical
    across-tuple standard deviation plus eps_eff.
    """
    x = np.asarray(effect, dtype=np.float64)
    tids = np.asarray([g[0] for g in groups], dtype=str)
    variants = np.asarray([g[1] for g in groups], dtype=str)
    if canonical_name not in set(variants):
        raise ValueError(f"carrier robustness requires a {canonical_name!r} realization")

    canonical_map: dict[str, np.ndarray] = {}
    for tid in sorted(set(tids[variants == canonical_name])):
        rows = (tids == tid) & (variants == canonical_name)
        canonical_map[tid] = x[rows].mean(axis=0)

    other_variants = sorted(set(variants) - {canonical_name})
    reference = next(iter(canonical_map.values()))
    if not other_variants:
        return np.ones_like(reference), np.zeros_like(reference)

    sign_matches = []
    standardized_changes = []
    for vv in other_variants:
        variant_tids = set(tids[variants == vv])
        common = sorted(set(canonical_map) & variant_tids)
        if not common:
            continue
        canonical_stack = np.stack([canonical_map[tid] for tid in common])
        variant_stack = []
        for tid in common:
            rows = (tids == tid) & (variants == vv)
            variant_stack.append(x[rows].mean(axis=0))
        variant_stack = np.stack(variant_stack)

        canonical_mean = canonical_stack.mean(axis=0)
        variant_mean = variant_stack.mean(axis=0)
        scale = canonical_stack.std(axis=0, ddof=1) if len(common) > 1 else np.zeros_like(canonical_mean)
        scale = scale + _effect_eps(canonical_stack)

        canonical_sign = np.sign(canonical_mean)
        variant_sign = np.sign(variant_mean)
        sign_matches.append((variant_sign == canonical_sign).astype(np.float64))
        standardized_changes.append(np.abs(variant_mean - canonical_mean) / np.maximum(scale, 1e-12))

    if not sign_matches:
        raise ValueError("no carrier variant has tuples matched to canonical realizations")
    return np.mean(sign_matches, axis=0), np.max(standardized_changes, axis=0)


def select_stable_heads(
    effect: np.ndarray,
    groups: list[tuple[str, str]],
    *,
    recurrence_min: float = DEFAULT_PROTOCOL.recurrence_min,
    sign_consistency_min: float = DEFAULT_PROTOCOL.sign_consistency_min,
    carrier_sensitivity_max: float = DEFAULT_PROTOCOL.carrier_sensitivity_max,
    n_resamples: int = DEFAULT_PROTOCOL.development_bootstrap_resamples,
    seed: int = 0,
    ci_level: float = DEFAULT_PROTOCOL.bootstrap_ci_level,
    require_positive_ci: bool = False,
    require_carrier_robustness: bool = True,
    max_heads: int | None = None,
) -> list[StableHead]:
    """Select stable positive heads using the stated localization protocol.

    Positivity is defined by a positive mean effect. Paired bootstrap intervals
    are reported for transparency but are not a deployment selection criterion
    unless ``require_positive_ci=True`` is explicitly requested.
    """
    x = np.asarray(effect, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("effect must have shape [G,L,H]")

    mean = x.mean(axis=0)
    recurrence, ci_low, ci_high = bootstrap_head_statistics(
        x, groups, n_resamples=n_resamples, seed=seed, ci_level=ci_level
    )
    if require_carrier_robustness:
        sign_consistency, carrier_sensitivity = carrier_robustness(x, groups)
    else:
        sign_consistency = np.ones_like(mean)
        carrier_sensitivity = np.zeros_like(mean)

    keep = (
        (mean > 0)
        & (recurrence >= recurrence_min)
        & (sign_consistency >= sign_consistency_min)
        & (carrier_sensitivity <= carrier_sensitivity_max)
    )
    if require_positive_ci:
        keep &= ci_low > 0

    order = np.argsort(np.where(keep, mean, -np.inf).ravel())[::-1]
    selected: list[StableHead] = []
    for flat in order:
        layer, head = np.unravel_index(flat, mean.shape)
        if not keep[layer, head]:
            break
        selected.append(
            StableHead(
                layer=int(layer),
                head=int(head),
                mean_effect=float(mean[layer, head]),
                ci_low=float(ci_low[layer, head]),
                ci_high=float(ci_high[layer, head]),
                recurrence=float(recurrence[layer, head]),
                sign_consistency=float(sign_consistency[layer, head]),
                carrier_sensitivity=float(carrier_sensitivity[layer, head]),
            )
        )
        if max_heads is not None and len(selected) >= max_heads:
            break
    return selected


def factorial_interaction_residuals(
    activations: np.ndarray,
    tuple_id: Iterable,
    source_level: Iterable,
    edit_level: Iterable,
    variant: Iterable | None = None,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Compute c_i^h=o11-o10-o01+o00 for every head."""
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("activations must have shape [N,L,H,Dh]")
    tids = np.asarray(list(tuple_id)).astype(str)
    s = np.asarray(list(source_level), dtype=int)
    a = np.asarray(list(edit_level), dtype=int)
    v = np.asarray(["canonical"] * len(tids) if variant is None else list(variant)).astype(str)

    residuals, groups = [], []
    for tid in sorted(set(tids)):
        for vv in sorted(set(v[tids == tid])):
            mask = (tids == tid) & (v == vv)
            cells = {}
            for sv in (0, 1):
                for av in (0, 1):
                    cell = mask & (s == sv) & (a == av)
                    if not np.any(cell):
                        raise ValueError(f"missing factorial cell ({sv},{av}) for tuple={tid}, variant={vv}")
                    cells[(sv, av)] = x[cell].mean(axis=0)
            residuals.append(cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)])
            groups.append((str(tid), str(vv)))
    return np.stack(residuals), groups
