#!/usr/bin/env python3
from __future__ import annotations

import argparse

from campie.hooks import attention_shape, discover_attention_output_projections
from campie.modeling import load_qwen_conditioner


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect the frozen multimodal conditioner attention stack.")
    p.add_argument("--model-id", default="Qwen/Qwen-Image-Edit")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()

    conditioner = load_qwen_conditioner(args.model_id, args.revision, args.dtype, args.device_map)
    layers, heads, head_dim = attention_shape(conditioner.model)
    projections = discover_attention_output_projections(conditioner.model)
    print(f"layers={layers}, heads/layer={heads}, head_dim={head_dim}, total_heads={layers * heads}")
    for item in projections:
        print(f"layer {item.layer:02d}: {item.name}")


if __name__ == "__main__":
    main()
