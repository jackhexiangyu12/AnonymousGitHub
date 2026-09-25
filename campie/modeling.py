from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor


class _ConditioningPassComplete(RuntimeError):
    """Internal control-flow sentinel used to stop the official edit pipeline after prompt encoding."""

    def __init__(self, encoded: Any):
        super().__init__("conditioning pass complete")
        self.encoded = encoded


@dataclass
class QwenConditioner:
    """Conditioning-only wrapper around the official Diffusers Qwen-Image-Edit pipeline.

    CAMP-IE does not define a separate prompt serializer or source-image resize rule here.
    The wrapper enters ``QwenImageEditPipeline.__call__`` and stops immediately after the
    pipeline's own ``encode_prompt`` finishes, before any VAE/DiT work is reached. This
    keeps the conditioner input path identical to the installed official pipeline while
    avoiding image generation during feature extraction.
    """

    pipeline: QwenImageEditPipeline
    model_id: str
    revision: str | None = None

    @property
    def model(self) -> Qwen2_5_VLForConditionalGeneration:
        return self.pipeline.text_encoder

    @property
    def processor(self) -> Qwen2VLProcessor:
        return self.pipeline.processor

    @torch.inference_mode()
    def encode(self, image: Image.Image, instruction: str) -> Any:
        """Run exactly the official Qwen-Image-Edit input/conditioning path, then stop.

        The official pipeline owns both text serialization and image preprocessing.
        We temporarily wrap ``encode_prompt`` only to terminate execution once the
        multimodal conditioner has completed; the returned object is the official
        prompt-encoding result. Attention hooks attached to ``self.model`` therefore
        observe the same conditioner forward pass used by the edit pipeline.
        """
        image = image.convert("RGB")
        original_encode_prompt = self.pipeline.encode_prompt

        def _stop_after_encode(*args: Any, **kwargs: Any) -> Any:
            encoded = original_encode_prompt(*args, **kwargs)
            raise _ConditioningPassComplete(encoded)

        self.pipeline.encode_prompt = _stop_after_encode  # type: ignore[method-assign]
        try:
            # Execution is intentionally intercepted after the official prompt encoder.
            # No VAE encoding, DiT denoising, or image generation is reached.
            self.pipeline(
                image=image,
                prompt=str(instruction),
                num_inference_steps=1,
                output_type="latent",
                return_dict=True,
            )
        except _ConditioningPassComplete as done:
            return done.encoded
        finally:
            self.pipeline.encode_prompt = original_encode_prompt  # type: ignore[method-assign]

        raise RuntimeError("QwenImageEditPipeline did not execute encode_prompt as expected")


def _parse_dtype(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _load_component(cls: Any, model_id: str, subfolder: str, common: dict[str, Any], **kwargs: Any) -> Any:
    """Load one component from the official Qwen-Image-Edit repository."""
    return cls.from_pretrained(model_id, subfolder=subfolder, **common, **kwargs)


def load_qwen_conditioner(
    model_id: str = "Qwen/Qwen-Image-Edit",
    revision: str | None = None,
    dtype: str = "bf16",
    device_map: str | None = "auto",
) -> QwenConditioner:
    """Load only the frozen Qwen multimodal conditioner, using official pipeline preprocessing.

    The heavyweight VAE and DiT are not loaded. A lightweight ``QwenImageEditPipeline``
    object is constructed from the checkpoint's official text encoder, processor, and
    tokenizer so that prompt serialization and source-image preprocessing are delegated
    to Diffusers rather than duplicated in CAMP-IE.
    """
    common: dict[str, Any] = {"revision": revision} if revision else {}
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _parse_dtype(dtype),
        "low_cpu_mem_usage": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    model = _load_component(
        Qwen2_5_VLForConditionalGeneration,
        model_id,
        "text_encoder",
        common,
        **model_kwargs,
    )
    processor = _load_component(Qwen2VLProcessor, model_id, "processor", common)
    tokenizer = _load_component(Qwen2Tokenizer, model_id, "tokenizer", common)

    # QwenImageEditPipeline.__init__ accepts registered components; the generation-only
    # components can remain absent because encode() stops immediately after encode_prompt.
    pipeline = QwenImageEditPipeline(
        scheduler=None,
        vae=None,
        text_encoder=model,
        tokenizer=tokenizer,
        processor=processor,
        transformer=None,
    )
    pipeline.set_progress_bar_config(disable=True)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return QwenConditioner(pipeline=pipeline, model_id=model_id, revision=revision)
