#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from campie.cis import fit_cis_monitor, save_cis_state
from campie.protocol import DEFAULT_PROTOCOL


def _head_pool(path: str) -> list[dict]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = report.get("CHH", [])
    if not pool:
        raise ValueError("head selection file contains no CHHs")
    return pool


def _heads_for_rho(pool: list[dict], rho: float) -> list[tuple[int, int]]:
    m = max(1, int(round(float(rho) * len(pool))))
    chosen = pool[:m]
    return [(int(item["layer"]), int(item["head"])) for item in chosen]


def _dev_log_loss_from_report(report: dict) -> float:
    losses = report["readout"]["development_log_loss"]
    selected = str(float(report["readout"]["selected_C"]))
    return float(losses[selected])


def main() -> None:
    p = argparse.ArgumentParser(description="Fit the core Contrastive Interaction Subspace from selected CHHs.")
    p.add_argument("--activations", required=True)
    p.add_argument("--head-selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--rho", default="auto",
        help="CHH pool fraction, or 'auto' to select from the paper's development grid.",
    )
    p.add_argument("--rank", default="auto", help="integer rank or 'auto' profile-likelihood elbow")
    p.add_argument(
        "--readout-c", default="auto",
        help="fixed logistic C or 'auto' to select from {0.1,1,10} on development request labels",
    )
    p.add_argument("--target-fpr", type=float, default=DEFAULT_PROTOCOL.target_fpr)
    args = p.parse_args()

    data = np.load(args.activations, allow_pickle=False)
    required = {
        "activations", "split", "tuple_id", "variant", "source_level", "edit_level", "risk_label"
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"NPZ missing fields needed for CIS fitting: {sorted(missing)}")

    split = data["split"].astype(str)
    tuple_id = data["tuple_id"].astype(str)
    variant = data["variant"].astype(str)
    source = data["source_level"].astype(int)
    edit = data["edit_level"].astype(int)
    label = data["risk_label"].astype(int)

    train = (split == "train") & (tuple_id != "")
    dev = split == "dev"
    if not np.any(train):
        raise ValueError("no training factorial rows found (split=train, non-empty tuple_id)")
    if not np.any(dev):
        raise ValueError("no development rows found")
    if set(np.unique(label[dev])) != {0, 1}:
        raise ValueError("development rows need both request-label classes for hyperparameter selection")

    pool = _head_pool(args.head_selection)
    rank = None if str(args.rank).lower() == "auto" else int(args.rank)
    c_value = None if str(args.readout_c).lower() == "auto" else float(args.readout_c)
    rho_values = DEFAULT_PROTOCOL.rho_grid if str(args.rho).lower() == "auto" else (float(args.rho),)

    candidates = []
    for rho in rho_values:
        selected = _heads_for_rho(pool, rho)
        state, report = fit_cis_monitor(
            train_activations=data["activations"][train],
            train_tuple_id=tuple_id[train],
            train_source_level=source[train],
            train_edit_level=edit[train],
            train_variant=variant[train],
            train_labels=label[train],
            selected_heads=selected,
            benign_train_mask=(label[train] == 0),
            dev_activations=data["activations"][dev],
            dev_labels=label[dev],
            benign_dev_mask=(label[dev] == 0),
            rank=rank,
            c_value=c_value,
            c_grid=DEFAULT_PROTOCOL.readout_c_grid,
            target_fpr=args.target_fpr,
            rho=float(rho),
        )
        candidates.append((_dev_log_loss_from_report(report), float(rho), state, report))

    # Development selection uses request-label log loss, distinct from the final
    # benign-FPR threshold calibration. Tie-break toward the smaller CHH pool.
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, selected_rho, state, report = candidates[0]
    report["development_selection"] = {
        "objective": "request-label logistic log loss",
        "rho_grid": [float(x) for x in rho_values],
        "selected_rho": float(selected_rho),
        "candidate_log_loss": {str(rho): float(loss) for loss, rho, _, _ in candidates},
    }
    report["data_counts"] = {
        "train_rows": int(np.sum(train)),
        "train_benign_rows": int(np.sum(train & (label == 0))),
        "train_harmful_rows": int(np.sum(train & (label == 1))),
        "dev_rows": int(np.sum(dev)),
        "dev_benign_rows": int(np.sum(dev & (label == 0))),
        "dev_harmful_rows": int(np.sum(dev & (label == 1))),
        "train_factorial_tuples": int(len(set(tuple_id[train]))),
        "train_tuple_variant_groups": int(len(set(zip(tuple_id[train], variant[train])))),
    }

    output = Path(args.output)
    save_cis_state(output, state)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"CHH pool={len(pool)} selected={len(state.selected_heads)} rho={state.rho:g}")
    print(f"CIS rank={state.basis.shape[1]} C={state.readout_c:g} threshold={state.threshold:.6g}")
    print(f"saved {output}")
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()
