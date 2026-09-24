#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from campie.hooks import HeadOutputRecorder, discover_attention_output_projections
from campie.modeling import load_qwen_conditioner


def main() -> None:
    p = argparse.ArgumentParser(description="Extract pre-WO Qwen conditioner head activations.")
    p.add_argument("--manifest", required=True, help="CSV containing image and instruction columns")
    p.add_argument("--image-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen-Image-Edit")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()

    df = pd.read_csv(args.manifest).fillna("")
    required = {"image", "instruction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if "sample_id" not in df:
        df["sample_id"] = [f"sample_{i:06d}" for i in range(len(df))]

    conditioner = load_qwen_conditioner(
        model_id=args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    projections = discover_attention_output_projections(conditioner.model)
    image_root = Path(args.image_root)
    activations = []

    with HeadOutputRecorder(conditioner.model, projections=projections, token_index=-1) as recorder:
        for idx, row in df.iterrows():
            image = Image.open(image_root / str(row.image)).convert("RGB")
            recorder.clear()
            conditioner.encode(image, str(row.instruction))
            act = recorder.numpy()
            if act.shape[0] != 1:
                raise RuntimeError("extractor expects one request at a time")
            activations.append(act[0])
            print(f"[{idx + 1}/{len(df)}] {row.sample_id}")

    payload: dict[str, np.ndarray] = {
        "activations": np.stack(activations).astype(np.float32),
        "sample_id": df.sample_id.astype(str).to_numpy(),
        "model_id": np.asarray(args.model_id),
        "revision": np.asarray(args.revision or ""),
    }
    for col in ["tuple_id", "variant", "direction_pair_id"]:
        if col in df:
            payload[col] = df[col].astype(str).to_numpy()
    for col in ["source_level", "edit_level", "direction_label"]:
        if col in df:
            payload[col] = df[col].astype(int).to_numpy()

    np.savez_compressed(args.output, **payload)
    print(f"saved {args.output}: activations={payload['activations'].shape}")


if __name__ == "__main__":
    main()
