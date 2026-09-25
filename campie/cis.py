from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .protocol import DEFAULT_PROTOCOL


HeadIndex = tuple[int, int]


@dataclass
class WhiteningState:
    mean: np.ndarray            # [d]
    basis: np.ndarray           # [d,r_b]
    covariance_eigs: np.ndarray # [r_b]
    eps: float
    sample_count: int = 0
    feature_dim: int = 0
    empirical_rank: int = 0
    covariance_trace: float = 0.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xc = x - self.mean
        base = 1.0 / np.sqrt(self.eps)
        if self.basis.size == 0:
            return xc * base
        proj = xc @ self.basis
        correction = (1.0 / np.sqrt(self.covariance_eigs + self.eps)) - base
        return xc * base + (proj * correction[None, :]) @ self.basis.T


@dataclass(frozen=True)
class CISBasisDiagnostics:
    contrast_count: int
    tuple_count: int
    tuple_variant_count: int
    feature_dim: int
    rank_e: int
    rank_s: int
    rank_c: int
    gamma: float
    gamma_fraction: float
    selected_rank: int
    positive_generalized_eigenvalues: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CISState:
    selected_heads: list[HeadIndex]
    whitener: WhiteningState
    basis: np.ndarray           # Q_C [d,r]
    eigenvalues: np.ndarray
    readout_weight: np.ndarray  # [r]
    readout_bias: float
    threshold: float
    readout_c: float = 1.0
    rho: float = 1.0
    diagnostics: CISBasisDiagnostics | None = None

    def coordinates(self, activations: np.ndarray) -> np.ndarray:
        g = gather_heads(activations, self.selected_heads)
        a = self.whitener.transform(g)
        return a @ self.basis

    def score(self, activations: np.ndarray) -> np.ndarray:
        q = self.coordinates(activations)
        return q @ self.readout_weight + self.readout_bias

    def block(self, activations: np.ndarray) -> np.ndarray:
        return self.score(activations) > self.threshold


def gather_heads(activations: np.ndarray, heads: Sequence[HeadIndex]) -> np.ndarray:
    """Concatenate selected pre-W_O head outputs ordered by layer then head."""
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("activations must have shape [N,L,H,Dh]")
    ordered = sorted((int(layer), int(head)) for layer, head in heads)
    if not ordered:
        raise ValueError("at least one selected head is required")
    pieces = [x[:, layer, head, :] for layer, head in ordered]
    return np.concatenate(pieces, axis=-1)


def fit_whitener(
    benign_features: np.ndarray,
    sample_weight: np.ndarray | None = None,
    eps_fraction: float = DEFAULT_PROTOCOL.whitening_eps_fraction,
) -> WhiteningState:
    """Fit W_B=(Sigma_B+eps_B I)^(-1/2) in compact SVD form."""
    x = np.asarray(benign_features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("benign_features must have shape [N,d] with N>=2")
    n, d = x.shape
    if sample_weight is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("invalid sample_weight")
        w = w / w.sum()

    mean = np.sum(x * w[:, None], axis=0)
    xc = x - mean
    weighted = xc * np.sqrt(w)[:, None]

    # Work in sample space because d_CIS can be much larger than the number of
    # benign training vectors. The nonzero covariance eigenvalues equal those
    # of weighted @ weighted.T.
    gram = weighted @ weighted.T
    eig, u = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eig)[::-1]
    eig = np.maximum(eig[order], 0.0)
    u = u[:, order]
    eig_max = float(eig.max()) if eig.size else 1.0
    keep = eig > max(np.finfo(np.float64).eps * eig_max, 1e-14)
    kept_eig = eig[keep]
    if np.any(keep):
        basis = weighted.T @ u[:, keep]
        basis = basis / np.sqrt(kept_eig)[None, :]
    else:
        basis = np.empty((d, 0), dtype=np.float64)
    trace = float(np.sum(eig))
    eps = max(float(eps_fraction * trace / max(d, 1)), 1e-8)
    return WhiteningState(
        mean=mean,
        basis=basis,
        covariance_eigs=kept_eig,
        eps=eps,
        sample_count=int(n),
        feature_dim=int(d),
        empirical_rank=int(np.sum(keep)),
        covariance_trace=trace,
    )


def equal_tuple_weights(tuple_id: Iterable, variant: Iterable | None = None) -> np.ndarray:
    """Give every tuple equal total weight, split equally across its rows."""
    tids = np.asarray(list(tuple_id)).astype(str)
    if variant is None:
        variant_arr = np.asarray(["canonical"] * len(tids), dtype=str)
    else:
        variant_arr = np.asarray(list(variant)).astype(str)
    if len(tids) != len(variant_arr):
        raise ValueError("metadata length mismatch")
    unique = sorted(set(tids))
    weight = np.zeros(len(tids), dtype=np.float64)
    for tid in unique:
        rows = np.where(tids == tid)[0]
        weight[rows] = 1.0 / (len(unique) * len(rows))
    return weight / weight.sum()


def vector_factorial_contrasts(
    features: np.ndarray,
    tuple_id: Iterable,
    source_level: Iterable,
    edit_level: Iterable,
    variant: Iterable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Vector E/S/C contrasts and weights 1/(n |V_i|)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must have shape [N,d]")
    tids = np.asarray(list(tuple_id)).astype(str)
    s = np.asarray(list(source_level), dtype=int)
    a = np.asarray(list(edit_level), dtype=int)
    v = np.asarray(["canonical"] * len(tids) if variant is None else list(variant)).astype(str)
    if not (len(tids) == len(s) == len(a) == len(v) == x.shape[0]):
        raise ValueError("metadata length mismatch")

    unique_tids = sorted(set(tids))
    edits, sources, interactions, weights, groups = [], [], [], [], []
    for tid in unique_tids:
        variants = sorted(set(v[tids == tid]))
        for vv in variants:
            mask = (tids == tid) & (v == vv)
            cells: dict[tuple[int, int], np.ndarray] = {}
            for sv in (0, 1):
                for av in (0, 1):
                    rows = mask & (s == sv) & (a == av)
                    if not np.any(rows):
                        raise ValueError(f"missing factorial cell ({sv},{av}) for tuple={tid}, variant={vv}")
                    cells[(sv, av)] = x[rows].mean(axis=0)
            x00, x01 = cells[(0, 0)], cells[(0, 1)]
            x10, x11 = cells[(1, 0)], cells[(1, 1)]
            edits.append(0.5 * ((x01 - x00) + (x11 - x10)))
            sources.append(0.5 * ((x10 - x00) + (x11 - x01)))
            interactions.append(x11 - x10 - x01 + x00)
            weights.append(1.0 / (len(unique_tids) * len(variants)))
            groups.append((str(tid), str(vv)))
    return np.stack(edits), np.stack(sources), np.stack(interactions), np.asarray(weights), groups


def _woodbury_solve(gamma: float, z: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve (gamma I + Z^T Z) X = rhs without a d x d inverse."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    rhs = np.asarray(rhs, dtype=np.float64)
    if z.size == 0:
        return rhs / gamma
    small = gamma * np.eye(z.shape[0], dtype=np.float64) + z @ z.T
    middle = np.linalg.solve(small, z @ rhs)
    return (rhs - z.T @ middle) / gamma


def profile_likelihood_rank(eigenvalues: np.ndarray, max_components: int = DEFAULT_PROTOCOL.max_scree_components) -> int:
    """Two-population profile-likelihood elbow on log eigenvalues."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    kmax = min(len(lam), int(max_components))
    if kmax <= 2:
        return max(1, kmax)
    y = np.log(lam[:kmax])
    best_k, best_ll = 1, -np.inf
    for k in range(1, kmax):
        left, right = y[:k], y[k:]
        mu1, mu2 = left.mean(), right.mean()
        ss = np.sum((left - mu1) ** 2) + np.sum((right - mu2) ** 2)
        var = max(ss / kmax, 1e-12)
        ll = -0.5 * kmax * np.log(var) - 0.5 * ss / var
        if ll > best_ll:
            best_k, best_ll = k, ll
    return int(best_k)


def fit_cis_basis(
    whitened_features: np.ndarray,
    tuple_id: Iterable,
    source_level: Iterable,
    edit_level: Iterable,
    variant: Iterable | None = None,
    rank: int | None = None,
    gamma_fraction: float = DEFAULT_PROTOCOL.gamma_fraction,
) -> tuple[np.ndarray, np.ndarray, CISBasisDiagnostics]:
    """Fit CIS from uncentered E/S/C second moments using a sample-space solve."""
    e, s, c, w, groups = vector_factorial_contrasts(
        whitened_features, tuple_id, source_level, edit_level, variant
    )
    sqrt_w = np.sqrt(w)[:, None]
    ew, sw, cw = e * sqrt_w, s * sqrt_w, c * sqrt_w
    d = whitened_features.shape[1]
    trace_es = float(np.sum(ew * ew) + np.sum(sw * sw))
    gamma = max(float(gamma_fraction * trace_es / max(d, 1)), 1e-8)
    z = np.concatenate([ew, sw], axis=0)

    binv_ct = _woodbury_solve(gamma, z, cw.T)
    kernel = cw @ binv_ct
    kernel = 0.5 * (kernel + kernel.T)
    evals, evecs = np.linalg.eigh(kernel)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    positive = evals > 1e-12
    evals = evals[positive]
    evecs = evecs[:, positive]
    if len(evals) == 0:
        raise RuntimeError("CIS generalized eigensystem has no positive eigenvalues")

    chosen_rank = profile_likelihood_rank(evals) if rank is None else int(rank)
    chosen_rank = max(1, min(chosen_rank, len(evals)))
    vr = evecs[:, :chosen_rank]
    lam = evals[:chosen_rank]
    ur = _woodbury_solve(gamma, z, cw.T @ vr) / np.sqrt(lam)[None, :]
    qc, _ = np.linalg.qr(ur, mode="reduced")

    tids = [g[0] for g in groups]
    def row_rank(m: np.ndarray) -> int:
        gram = m @ m.T
        eig = np.linalg.eigvalsh(0.5 * (gram + gram.T))
        if eig.size == 0:
            return 0
        tol = max(np.finfo(np.float64).eps * max(m.shape) * float(np.max(eig)), 1e-12)
        return int(np.sum(eig > tol))

    diagnostics = CISBasisDiagnostics(
        contrast_count=int(len(groups)),
        tuple_count=int(len(set(tids))),
        tuple_variant_count=int(len(groups)),
        feature_dim=int(d),
        rank_e=row_rank(ew),
        rank_s=row_rank(sw),
        rank_c=row_rank(cw),
        gamma=float(gamma),
        gamma_fraction=float(gamma_fraction),
        selected_rank=int(chosen_rank),
        positive_generalized_eigenvalues=int(len(evals)),
    )
    return qc[:, :chosen_rank], evals, diagnostics


def fit_linear_readout(
    coordinates: np.ndarray,
    labels: np.ndarray,
    c_value: float = 1.0,
) -> tuple[np.ndarray, float]:
    """L2 logistic readout fit from request labels, not generated-output scores."""
    q = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if q.ndim != 2 or y.shape != (q.shape[0],):
        raise ValueError("coordinate/label shape mismatch")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be binary")
    if len(np.unique(y)) != 2:
        raise ValueError("both request-label classes are required")
    clf = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000)
    clf.fit(q, y)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def _binary_log_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, s) - y * s))


def select_linear_readout(
    train_coordinates: np.ndarray,
    train_labels: np.ndarray,
    dev_coordinates: np.ndarray,
    dev_labels: np.ndarray,
    c_grid: Sequence[float] = DEFAULT_PROTOCOL.readout_c_grid,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    """Choose L2 strength on development request-label log loss."""
    losses: dict[str, float] = {}
    candidates = []
    for c_value in c_grid:
        weight, bias = fit_linear_readout(train_coordinates, train_labels, c_value=float(c_value))
        loss = _binary_log_loss(dev_coordinates @ weight + bias, dev_labels)
        losses[str(float(c_value))] = loss
        candidates.append((loss, float(c_value), weight, bias))
    # deterministic tie-break: stronger regularization (smaller C)
    candidates.sort(key=lambda t: (t[0], t[1]))
    _, selected_c, weight, bias = candidates[0]
    return weight, bias, selected_c, losses


def calibrate_threshold(benign_scores: np.ndarray, target_fpr: float = DEFAULT_PROTOCOL.target_fpr) -> float:
    """Choose a conservative development threshold with empirical FPR <= target."""
    scores = np.sort(np.asarray(benign_scores, dtype=np.float64))[::-1]
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("benign_scores must be a non-empty vector")
    max_block = int(np.floor(target_fpr * len(scores)))
    if max_block <= 0:
        return float(np.inf)
    if max_block >= len(scores):
        return float(-np.inf)
    return float(scores[max_block])


def signed_coordinate_summary(
    coordinates: np.ndarray,
    tuple_id: Iterable,
    source_level: Iterable,
    edit_level: Iterable,
    variant: Iterable | None = None,
) -> dict:
    """Summarize signed CIS coordinates without computing benchmark metrics."""
    q = np.asarray(coordinates, dtype=np.float64)
    tids = np.asarray(list(tuple_id)).astype(str)
    s = np.asarray(list(source_level), dtype=int)
    a = np.asarray(list(edit_level), dtype=int)
    v = np.asarray(["canonical"] * len(tids) if variant is None else list(variant)).astype(str)
    if q.ndim != 2 or not (len(tids) == len(s) == len(a) == len(v) == len(q)):
        raise ValueError("coordinate/metadata shape mismatch")

    cells = {}
    for sv in (0, 1):
        for av in (0, 1):
            rows = (s == sv) & (a == av)
            values = q[rows]
            cells[f"S{sv}A{av}"] = {
                "n": int(len(values)),
                "mean": values.mean(axis=0).tolist() if len(values) else [],
                "std": values.std(axis=0, ddof=1).tolist() if len(values) > 1 else [0.0] * q.shape[1],
            }

    _, _, c, _, _ = vector_factorial_contrasts(q, tids, s, a, v)
    return {
        "cells": cells,
        "interaction": {
            "n": int(len(c)),
            "mean": c.mean(axis=0).tolist(),
            "std": c.std(axis=0, ddof=1).tolist() if len(c) > 1 else [0.0] * q.shape[1],
        },
    }


def fit_cis_monitor(
    train_activations: np.ndarray,
    train_tuple_id: Iterable,
    train_source_level: Iterable,
    train_edit_level: Iterable,
    train_variant: Iterable,
    train_labels: np.ndarray,
    selected_heads: Sequence[HeadIndex],
    benign_train_mask: np.ndarray,
    dev_activations: np.ndarray,
    dev_labels: np.ndarray,
    benign_dev_mask: np.ndarray,
    *,
    rank: int | None = None,
    c_value: float | None = None,
    c_grid: Sequence[float] = DEFAULT_PROTOCOL.readout_c_grid,
    target_fpr: float = DEFAULT_PROTOCOL.target_fpr,
    whitening_eps_fraction: float = DEFAULT_PROTOCOL.whitening_eps_fraction,
    gamma_fraction: float = DEFAULT_PROTOCOL.gamma_fraction,
    rho: float = 1.0,
) -> tuple[CISState, dict]:
    """Fit whitening, CIS basis, readout, and development threshold."""
    selected = sorted((int(l), int(h)) for l, h in selected_heads)
    train_g = gather_heads(train_activations, selected)
    benign_train_mask = np.asarray(benign_train_mask, dtype=bool)
    if benign_train_mask.shape != (len(train_g),):
        raise ValueError("benign_train_mask shape mismatch")

    tids = np.asarray(list(train_tuple_id)).astype(str)
    variants = np.asarray(list(train_variant)).astype(str)
    benign_weights = equal_tuple_weights(tids[benign_train_mask], variants[benign_train_mask])
    whitener = fit_whitener(
        train_g[benign_train_mask], sample_weight=benign_weights, eps_fraction=whitening_eps_fraction
    )
    train_a = whitener.transform(train_g)
    basis, eigenvalues, diagnostics = fit_cis_basis(
        train_a,
        train_tuple_id,
        train_source_level,
        train_edit_level,
        train_variant,
        rank=rank,
        gamma_fraction=gamma_fraction,
    )
    q_train = train_a @ basis

    dev_g = gather_heads(dev_activations, selected)
    q_dev = whitener.transform(dev_g) @ basis
    train_labels = np.asarray(train_labels, dtype=int)
    dev_labels = np.asarray(dev_labels, dtype=int)
    if c_value is None:
        weight, bias, selected_c, c_losses = select_linear_readout(
            q_train, train_labels, q_dev, dev_labels, c_grid=c_grid
        )
    else:
        weight, bias = fit_linear_readout(q_train, train_labels, c_value=float(c_value))
        selected_c = float(c_value)
        c_losses = {str(selected_c): _binary_log_loss(q_dev @ weight + bias, dev_labels)}

    dev_scores = q_dev @ weight + bias
    benign_dev_mask = np.asarray(benign_dev_mask, dtype=bool)
    threshold = calibrate_threshold(dev_scores[benign_dev_mask], target_fpr=target_fpr)
    state = CISState(
        selected_heads=selected,
        whitener=whitener,
        basis=basis,
        eigenvalues=eigenvalues,
        readout_weight=weight,
        readout_bias=bias,
        threshold=threshold,
        readout_c=selected_c,
        rho=float(rho),
        diagnostics=diagnostics,
    )
    fit_report = {
        "rho": float(rho),
        "selected_heads": int(len(selected)),
        "head_width": int(train_activations.shape[-1]),
        "d_cis": int(train_g.shape[1]),
        "whitening": {
            "benign_vectors": int(whitener.sample_count),
            "feature_dim": int(whitener.feature_dim),
            "empirical_rank_before_regularization": int(whitener.empirical_rank),
            "covariance_trace": float(whitener.covariance_trace),
            "epsilon_B": float(whitener.eps),
            "epsilon_fraction": float(whitening_eps_fraction),
        },
        "cis": diagnostics.to_dict(),
        "readout": {
            "selected_C": float(selected_c),
            "C_grid": [float(x) for x in c_grid],
            "development_log_loss": c_losses,
            "threshold": float(threshold),
            "target_benign_fpr": float(target_fpr),
        },
        "signed_coordinate_summary_train": signed_coordinate_summary(
            q_train, train_tuple_id, train_source_level, train_edit_level, train_variant
        ),
    }
    return state, fit_report


def save_cis_state(path: str | Path, state: CISState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    diag = state.diagnostics
    np.savez_compressed(
        path,
        selected_heads=np.asarray(state.selected_heads, dtype=np.int32),
        whitening_mean=state.whitener.mean.astype(np.float32),
        whitening_basis=state.whitener.basis.astype(np.float32),
        whitening_eigs=state.whitener.covariance_eigs.astype(np.float32),
        whitening_eps=np.asarray(state.whitener.eps, dtype=np.float32),
        whitening_sample_count=np.asarray(state.whitener.sample_count, dtype=np.int32),
        whitening_feature_dim=np.asarray(state.whitener.feature_dim, dtype=np.int32),
        whitening_empirical_rank=np.asarray(state.whitener.empirical_rank, dtype=np.int32),
        whitening_trace=np.asarray(state.whitener.covariance_trace, dtype=np.float64),
        cis_basis=state.basis.astype(np.float32),
        cis_eigenvalues=state.eigenvalues.astype(np.float32),
        readout_weight=state.readout_weight.astype(np.float32),
        readout_bias=np.asarray(state.readout_bias, dtype=np.float32),
        threshold=np.asarray(state.threshold, dtype=np.float32),
        readout_c=np.asarray(state.readout_c, dtype=np.float32),
        rho=np.asarray(state.rho, dtype=np.float32),
        cis_diagnostics=np.asarray([] if diag is None else [
            diag.contrast_count, diag.tuple_count, diag.tuple_variant_count, diag.feature_dim,
            diag.rank_e, diag.rank_s, diag.rank_c, diag.selected_rank,
            diag.positive_generalized_eigenvalues,
        ], dtype=np.float64),
        cis_gamma=np.asarray(np.nan if diag is None else diag.gamma, dtype=np.float64),
        cis_gamma_fraction=np.asarray(np.nan if diag is None else diag.gamma_fraction, dtype=np.float64),
    )


def load_cis_state(path: str | Path) -> CISState:
    data = np.load(path, allow_pickle=False)
    whitener = WhiteningState(
        mean=data["whitening_mean"].astype(np.float64),
        basis=data["whitening_basis"].astype(np.float64),
        covariance_eigs=data["whitening_eigs"].astype(np.float64),
        eps=float(data["whitening_eps"]),
        sample_count=int(data["whitening_sample_count"]) if "whitening_sample_count" in data else 0,
        feature_dim=int(data["whitening_feature_dim"]) if "whitening_feature_dim" in data else len(data["whitening_mean"]),
        empirical_rank=int(data["whitening_empirical_rank"]) if "whitening_empirical_rank" in data else len(data["whitening_eigs"]),
        covariance_trace=float(data["whitening_trace"]) if "whitening_trace" in data else float(np.sum(data["whitening_eigs"])),
    )
    diagnostics = None
    if "cis_diagnostics" in data and len(data["cis_diagnostics"]):
        v = data["cis_diagnostics"].astype(int)
        diagnostics = CISBasisDiagnostics(
            contrast_count=int(v[0]), tuple_count=int(v[1]), tuple_variant_count=int(v[2]),
            feature_dim=int(v[3]), rank_e=int(v[4]), rank_s=int(v[5]), rank_c=int(v[6]),
            gamma=float(data["cis_gamma"]), gamma_fraction=float(data["cis_gamma_fraction"]),
            selected_rank=int(v[7]), positive_generalized_eigenvalues=int(v[8]),
        )
    return CISState(
        selected_heads=[tuple(map(int, row)) for row in data["selected_heads"]],
        whitener=whitener,
        basis=data["cis_basis"].astype(np.float64),
        eigenvalues=data["cis_eigenvalues"].astype(np.float64),
        readout_weight=data["readout_weight"].astype(np.float64),
        readout_bias=float(data["readout_bias"]),
        threshold=float(data["threshold"]),
        readout_c=float(data["readout_c"]) if "readout_c" in data else 1.0,
        rho=float(data["rho"]) if "rho" in data else 1.0,
        diagnostics=diagnostics,
    )
