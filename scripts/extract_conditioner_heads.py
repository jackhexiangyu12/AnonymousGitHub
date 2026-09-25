#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import diffusers
import transformers
from PIL import Image

from campie.data import manifest_summary, validate_manifest
from campie.hooks import HeadOutputRecorder, attention_shape, discover_attention_output_projections
from campie.modeling import load_qwen_conditioner


_STRING_COLUMNS = [
    "tuple_id", "variant", "direction_pair_id", "split", "source_group",
    "normalized_edit_template",
]
_INT_COLUMNS = ["source_level", "edit_level", "direction_label", "risk_label"]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract final-token pre-W_O conditioner head activations.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen-Image-Edit")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()

    df = pd.read_csv(args.manifest).fillna("")
    validate_manifest(df)
    if "sample_id" not in df:
        df["sample_id"] = [f"sample_{i:06d}" for i in range(len(df))]

    conditioner = load_qwen_conditioner(
        model_id=args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    projections = discover_attention_output_projections(conditioner.model)
    num_layers, num_heads, head_dim = attention_shape(conditioner.model)
    summary = manifest_summary(df)
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
        "conditioning_backend": np.asarray("diffusers.QwenImageEditPipeline"),
        "diffusers_version": np.asarray(diffusers.__version__),
        "transformers_version": np.asarray(transformers.__version__),
        "num_layers": np.asarray(num_layers, dtype=np.int32),
        "num_heads": np.asarray(num_heads, dtype=np.int32),
        "head_dim": np.asarray(head_dim, dtype=np.int32),
        "manifest_rows": np.asarray(summary.rows, dtype=np.int32),
        "direction_pairs": np.asarray(summary.direction_pairs, dtype=np.int32),
        "factorial_tuples": np.asarray(summary.factorial_tuples, dtype=np.int32),
        "factorial_tuple_variants": np.asarray(summary.factorial_tuple_variants, dtype=np.int32),
    }
    for col in _STRING_COLUMNS:
        if col in df:
            payload[col] = df[col].astype(str).to_numpy()
    for col in _INT_COLUMNS:
        if col in df:
            payload[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).to_numpy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved {args.output}: activations={payload['activations'].shape}")


if __name__ == "__main__":
    main()
