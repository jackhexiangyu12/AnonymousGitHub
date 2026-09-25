#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from campie.cis import load_cis_state
from campie.hooks import HeadOutputRecorder, discover_attention_output_projections
from campie.modeling import load_qwen_conditioner


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one ordinary multimodal conditioning pass and return the pre-generation CAMP-IE decision."
    )
    p.add_argument("--image", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen-Image-Edit")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()

    state = load_cis_state(args.state)
    conditioner = load_qwen_conditioner(
        model_id=args.model_id, revision=args.revision, dtype=args.dtype, device_map=args.device_map
    )
    projections = discover_attention_output_projections(conditioner.model)
    image = Image.open(args.image).convert("RGB")

    with HeadOutputRecorder(conditioner.model, projections=projections, token_index=-1) as recorder:
        conditioner.encode(image, args.instruction)
        activation = recorder.numpy()

    score = float(state.score(activation)[0])
    block = bool(score > state.threshold)
    print(f"risk_score={score:.8g}")
    print(f"threshold={state.threshold:.8g}")
    print("decision=BLOCK" if block else "decision=ALLOW")
    # This script intentionally stops here. In the full image-editing system an
    # ALLOW decision forwards the original, unmodified conditioning state to the
    # generator; a BLOCK decision terminates before DiT denoising.


if __name__ == "__main__":
    main()
