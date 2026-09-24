#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from campie.head_analysis import (
    factorial_head_effects,
    fit_harmfulness_direction,
    project_head_scores,
    rank_positive_heads,
)


def _paired_direction_rows(data) -> tuple[np.ndarray, np.ndarray]:
    required = {"direction_pair_id", "direction_label"}
    if not required.issubset(data.files):
        raise ValueError("NPZ must contain direction_pair_id and direction_label for direction calibration")
    pair_id = data["direction_pair_id"].astype(str)
    label = data["direction_label"].astype(int)
    acts = data["activations"]

    benign, harmful = [], []
    for pid in sorted(set(pair_id) - {""}):
        b = np.where((pair_id == pid) & (label == 0))[0]
        h = np.where((pair_id == pid) & (label == 1))[0]
        if len(b) != 1 or len(h) != 1:
            raise ValueError(f"direction pair {pid!r} must contain exactly one benign and one harmful row")
        benign.append(acts[b[0]])
        harmful.append(acts[h[0]])
    if not benign:
        raise ValueError("no complete direction calibration pairs found")
    return np.stack(harmful), np.stack(benign)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute EHH/SHH/CHH head effects from extracted activations.")
    p.add_argument("--activations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=72)
    args = p.parse_args()

    data = np.load(args.activations, allow_pickle=False)
    required = {"activations", "tuple_id", "source_level", "edit_level"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"NPZ missing fields: {sorted(missing)}")

    harmful, benign = _paired_direction_rows(data)
    direction = fit_harmfulness_direction(harmful, benign)
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

    ranked = {
        "EHH": rank_positive_heads(effects.edit, top_k=args.top_k),
        "SHH": rank_positive_heads(effects.source, top_k=args.top_k),
        "CHH": rank_positive_heads(effects.interaction, top_k=args.top_k),
    }
    (out / "ranked_heads.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print(f"saved {out / 'head_effects.npz'}")
    print(f"saved {out / 'ranked_heads.json'}")


if __name__ == "__main__":
    main()
