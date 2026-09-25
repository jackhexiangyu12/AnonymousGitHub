#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from campie.head_analysis import (
    factorial_head_effects,
    fit_layer_harmfulness_direction,
    project_head_scores,
    select_stable_heads,
)
from campie.protocol import DEFAULT_PROTOCOL


def _paired_direction_rows(data) -> tuple[np.ndarray, np.ndarray, int]:
    required = {"direction_pair_id", "direction_label"}
    if not required.issubset(data.files):
        raise ValueError("NPZ must contain direction_pair_id and direction_label")
    pair_id = data["direction_pair_id"].astype(str)
    label = data["direction_label"].astype(int)
    acts = data["activations"]

    benign, harmful = [], []
    complete = sorted(set(pair_id) - {""})
    for pid in complete:
        b = np.where((pair_id == pid) & (label == 0))[0]
        h = np.where((pair_id == pid) & (label == 1))[0]
        if len(b) != 1 or len(h) != 1:
            raise ValueError(f"direction pair {pid!r} must contain one benign and one harmful row")
        benign.append(acts[b[0]])
        harmful.append(acts[h[0]])
    if not benign:
        raise ValueError("no complete harmfulness-direction calibration pairs found")
    return np.stack(harmful), np.stack(benign), len(complete)


def main() -> None:
    p = argparse.ArgumentParser(description="Localize stable EHH/SHH/CHH heads from matched 2x2 tuples.")
    p.add_argument("--activations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_PROTOCOL.bootstrap_resamples)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recurrence-min", type=float, default=DEFAULT_PROTOCOL.recurrence_min)
    p.add_argument("--sign-consistency-min", type=float, default=DEFAULT_PROTOCOL.sign_consistency_min)
    p.add_argument("--carrier-sensitivity-max", type=float, default=DEFAULT_PROTOCOL.carrier_sensitivity_max)
    p.add_argument("--ci-level", type=float, default=DEFAULT_PROTOCOL.bootstrap_ci_level)
    p.add_argument(
        "--require-positive-ci",
        action="store_true",
        help="Optional stricter mode. The paper protocol uses positive mean + recurrence; CIs are descriptive by default.",
    )
    args = p.parse_args()

    data = np.load(args.activations, allow_pickle=False)
    required = {"activations", "tuple_id", "source_level", "edit_level"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"NPZ missing fields: {sorted(missing)}")

    harmful, benign, n_direction_pairs = _paired_direction_rows(data)
    direction = fit_layer_harmfulness_direction(harmful, benign)
    scores = project_head_scores(data["activations"], direction)

    tuple_id = data["tuple_id"].astype(str)
    use = tuple_id != ""
    variants = data["variant"].astype(str)[use] if "variant" in data.files else None
    effects = factorial_head_effects(
        scores=scores[use],
        tuple_id=tuple_id[use],
        source_level=data["source_level"][use],
        edit_level=data["edit_level"][use],
        variant=variants,
    )

    common = dict(
        recurrence_min=args.recurrence_min,
        sign_consistency_min=args.sign_consistency_min,
        carrier_sensitivity_max=args.carrier_sensitivity_max,
        n_resamples=args.bootstrap,
        seed=args.seed,
        ci_level=args.ci_level,
        require_positive_ci=args.require_positive_ci,
    )
    ehh = select_stable_heads(effects.edit, effects.groups, require_carrier_robustness=False, **common)
    shh = select_stable_heads(effects.source, effects.groups, require_carrier_robustness=False, **common)
    chh = select_stable_heads(effects.interaction, effects.groups, require_carrier_robustness=True, **common)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "head_effects.npz",
        edit=effects.edit.astype(np.float32),
        source=effects.source.astype(np.float32),
        interaction=effects.interaction.astype(np.float32),
        harmfulness_direction=direction.direction.astype(np.float32),
        benign_mean=direction.benign_mean.astype(np.float32),
        benign_std=direction.benign_std.astype(np.float32),
        eps=np.asarray(direction.eps, dtype=np.float32),
    )
    variants_present = sorted({g[1] for g in effects.groups})
    report = {
        "harmfulness_direction": {
            "estimator": "paired mean difference after within-layer head averaging",
            "scope": "one direction per layer; benign normalization remains head-specific",
            "calibration_pairs": int(n_direction_pairs),
            "eps_z": float(direction.eps),
        },
        "selection": {
            "positivity_definition": "mean paired standardized effect > 0",
            "bootstrap_unit": "matched tuple/source group",
            "bootstrap_resamples": int(args.bootstrap),
            "bootstrap_ci_level": float(args.ci_level),
            "ci_multiplicity_correction": DEFAULT_PROTOCOL.ci_multiplicity,
            "ci_used_for_selection": bool(args.require_positive_ci),
            "recurrence_min": float(args.recurrence_min),
            "sign_consistency_min": float(args.sign_consistency_min),
            "carrier_sensitivity_max": float(args.carrier_sensitivity_max),
            "carrier_variants": variants_present,
            "factorial_tuple_variant_groups": int(len(effects.groups)),
            "factorial_tuples": int(len({g[0] for g in effects.groups})),
        },
        "EHH": [h.to_dict() for h in ehh],
        "SHH": [h.to_dict() for h in shh],
        "CHH": [h.to_dict() for h in chh],
    }
    (out / "head_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"EHH={len(ehh)} SHH={len(shh)} CHH={len(chh)}")
    print(f"saved {out / 'head_effects.npz'}")
    print(f"saved {out / 'head_selection.json'}")


if __name__ == "__main__":
    main()
