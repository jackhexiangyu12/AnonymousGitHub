from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor


# Prompt format used by the Qwen-Image-Edit conditioner. Keeping the conditioner
# input format aligned with the official pipeline matters for head-level analysis.
_QWEN_EDIT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class QwenConditioner:
    model: Qwen2_5_VLForConditionalGeneration
    processor: Qwen2VLProcessor
    model_id: str
    revision: str | None = None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def prepare_inputs(self, image: Image.Image, instruction: str) -> dict[str, torch.Tensor]:
        image = resize_condition_image(image)
        text = _QWEN_EDIT_TEMPLATE.format(str(instruction))
        batch = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        # With accelerate/device_map, sending inputs to the embedding device is the
        # safest default. For a single-device model this is simply model.device.
        input_device = _input_device(self.model)
        return {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in batch.items()}

    @torch.inference_mode()
    def encode(self, image: Image.Image, instruction: str) -> Any:
        inputs = self.prepare_inputs(image, instruction)
        return self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )


def _input_device(model: torch.nn.Module) -> torch.device:
    # The token embedding is a reliable entry point when a device map is active.
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def resize_condition_image(image: Image.Image, target_area: int = 1024 * 1024, multiple: int = 32) -> Image.Image:
    """Resize to the same ~1MP, multiple-of-32 geometry used by Qwen-Image-Edit."""
    image = image.convert("RGB")
    w, h = image.size
    ratio = w / max(h, 1)
    raw_w = math.sqrt(target_area * ratio)
    raw_h = raw_w / ratio
    new_w = max(multiple, round(raw_w / multiple) * multiple)
    new_h = max(multiple, round(raw_h / multiple) * multiple)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _parse_dtype(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[key]


def load_qwen_conditioner(
    model_id: str = "Qwen/Qwen-Image-Edit",
    revision: str | None = None,
    dtype: str = "bf16",
    device_map: str | None = "auto",
) -> QwenConditioner:
    """Load only the frozen Qwen2.5-VL conditioner and its processor.

    This intentionally avoids loading the DiT and VAE because CAMP-IE's head
    localization is performed before image generation.
    """
    common: dict[str, Any] = {"revision": revision} if revision else {}
    model_kwargs: dict[str, Any] = {
        **common,
        "subfolder": "text_encoder",
        "torch_dtype": _parse_dtype(dtype),
        "low_cpu_mem_usage": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)

    try:
        processor = Qwen2VLProcessor.from_pretrained(model_id, subfolder="processor", **common)
    except OSError:
        # Compatibility fallback for checkpoints that store processor files at root.
        processor = Qwen2VLProcessor.from_pretrained(model_id, **common)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return QwenConditioner(model=model, processor=processor, model_id=model_id, revision=revision)
