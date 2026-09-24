from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class HeadDirection:
    direction: np.ndarray   # [L,H,Dh]
    benign_mean: np.ndarray # [L,H]
    benign_std: np.ndarray  # [L,H]
    eps: float


@dataclass
class FactorialHeadEffects:
    edit: np.ndarray        # [G,L,H]
    source: np.ndarray      # [G,L,H]
    interaction: np.ndarray # [G,L,H]
    groups: list[tuple[str, str]]


def fit_harmfulness_direction(
    harmful: np.ndarray,
    benign: np.ndarray,
    eps_scale: float = 1e-4,
) -> HeadDirection:
    """Fit one harmfulness direction per layer/head from matched calibration pairs.

    ``harmful`` and ``benign`` are ``[N,L,H,Dh]`` with pair-aligned first axes.
    The direction is the normalized mean paired difference. Standardization uses
    the benign projections only.
    """
    h = np.asarray(harmful, dtype=np.float64)
    b = np.asarray(benign, dtype=np.float64)
    if h.shape != b.shape or h.ndim != 4:
        raise ValueError("harmful and benign must have identical shape [N,L,H,Dh]")

    direction = np.mean(h - b, axis=0)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-12)

    benign_raw = np.einsum("nlhd,lhd->nlh", b, direction)
    mu = benign_raw.mean(axis=0)
    sigma = benign_raw.std(axis=0, ddof=1)
    positive = sigma[sigma > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = float(eps_scale * scale)
    return HeadDirection(direction=direction, benign_mean=mu, benign_std=sigma, eps=eps)


def project_head_scores(activations: np.ndarray, state: HeadDirection) -> np.ndarray:
    """Project activations onto harmfulness directions and benign-standardize."""
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:] != state.direction.shape:
        raise ValueError(f"expected [N,{','.join(map(str, state.direction.shape))}], got {x.shape}")
    raw = np.einsum("nlhd,lhd->nlh", x, state.direction)
    return (raw - state.benign_mean[None]) / (state.benign_std[None] + state.eps)


def factorial_head_effects(
    scores: np.ndarray,
    tuple_id: Iterable,
    source_level: Iterable,
    edit_level: Iterable,
    variant: Iterable | None = None,
) -> FactorialHeadEffects:
    """Compute EHH, SHH and CHH factorial contrasts for every head.

    For each matched 2x2 group:
      E = 1/2[(z01-z00) + (z11-z10)]
      S = 1/2[(z10-z00) + (z11-z01)]
      C = z11-z10-z01+z00
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
    for tid in np.unique(tids):
        for vv in np.unique(v[tids == tid]):
            mask = (tids == tid) & (v == vv)
            cells = {}
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

    return FactorialHeadEffects(
        edit=np.stack(edits),
        source=np.stack(sources),
        interaction=np.stack(interactions),
        groups=groups,
    )


def rank_positive_heads(effect: np.ndarray, top_k: int = 72) -> list[dict[str, float | int]]:
    """Rank heads by their mean positive paired effect across matched groups."""
    x = np.asarray(effect, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("effect must have shape [G,L,H]")
    mean = x.mean(axis=0)
    positive = np.maximum(mean, 0.0)
    order = np.argsort(positive.ravel())[::-1]
    result: list[dict[str, float | int]] = []
    for flat in order:
        layer, head = np.unravel_index(flat, positive.shape)
        if positive[layer, head] <= 0 or len(result) >= top_k:
            break
        # sign rate is useful for inspecting whether the effect is consistently positive.
        sign_rate = float(np.mean(x[:, layer, head] > 0))
        result.append(
            {
                "layer": int(layer),
                "head": int(head),
                "mean_effect": float(mean[layer, head]),
                "positive_fraction": sign_rate,
            }
        )
    return result
