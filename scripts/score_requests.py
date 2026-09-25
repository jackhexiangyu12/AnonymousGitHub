#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from campie.cis import load_cis_state


def main() -> None:
    p = argparse.ArgumentParser(description="Apply the pre-generation CAMP-IE readout to extracted activations.")
    p.add_argument("--activations", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = np.load(args.activations, allow_pickle=False)
    state = load_cis_state(args.state)
    scores = state.score(data["activations"])
    blocked = scores > state.threshold
    sample_id = data["sample_id"].astype(str) if "sample_id" in data.files else np.arange(len(scores)).astype(str)
    out = pd.DataFrame({"sample_id": sample_id, "risk_score": scores, "block": blocked.astype(int)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
