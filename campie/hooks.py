from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class AttentionOutputProjection:
    layer: int
    name: str
    module: torch.nn.Module


def _text_config(model: torch.nn.Module):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def attention_shape(model: torch.nn.Module) -> tuple[int, int, int]:
    cfg = _text_config(model)
    num_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    return num_layers, num_heads, head_dim


def discover_attention_output_projections(model: torch.nn.Module) -> list[AttentionOutputProjection]:
    """Find Qwen2.5-VL language self-attention output projections.

    We hook the input of ``o_proj`` so the captured tensor is the concatenated
    per-head attention output before the output projection W_O.
    """
    num_layers, _, _ = attention_shape(model)
    pattern = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn\.o_proj$")
    found: dict[int, AttentionOutputProjection] = {}

    for name, module in model.named_modules():
        m = pattern.search(name)
        if m:
            layer = int(m.group(1))
            found[layer] = AttentionOutputProjection(layer=layer, name=name, module=module)

    # Fallback for transformers naming changes while still excluding the vision tower.
    if len(found) != num_layers:
        fallback = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")
        for name, module in model.named_modules():
            if "visual" in name or "vision" in name:
                continue
            m = fallback.search(name)
            if m:
                layer = int(m.group(1))
                found.setdefault(layer, AttentionOutputProjection(layer=layer, name=name, module=module))

    missing = [i for i in range(num_layers) if i not in found]
    if missing:
        examples = [n for n, _ in list(model.named_modules()) if n.endswith("self_attn.o_proj")][:8]
        raise RuntimeError(
            f"could not resolve all language attention layers; missing={missing}; examples={examples}"
        )
    return [found[i] for i in range(num_layers)]


class HeadOutputRecorder:
    """Record a selected token from every attention head before ``o_proj``.

    Returned activations have shape ``[batch, layer, head, head_dim]``.
    The paper-level analysis uses the final conditioning token, so ``token_index``
    defaults to ``-1``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        projections: Iterable[AttentionOutputProjection] | None = None,
        token_index: int = -1,
    ) -> None:
        self.model = model
        self.projections = list(projections or discover_attention_output_projections(model))
        self.token_index = int(token_index)
        self.num_layers, self.num_heads, self.head_dim = attention_shape(model)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._cache: dict[int, torch.Tensor] = {}

    def _pre_hook(self, layer: int):
        def hook(_module, args):
            x = args[0]
            if x.ndim != 3:
                raise RuntimeError(f"expected [B,T,D] before o_proj, got {tuple(x.shape)}")
            expected = self.num_heads * self.head_dim
            if x.shape[-1] != expected:
                raise RuntimeError(f"attention width {x.shape[-1]} != num_heads*head_dim ({expected})")
            token = x[:, self.token_index, :].detach().float().cpu()
            self._cache[layer] = token.reshape(token.shape[0], self.num_heads, self.head_dim)
        return hook

    def __enter__(self):
        self.clear()
        for item in self.projections:
            self._handles.append(item.module.register_forward_pre_hook(self._pre_hook(item.layer)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self._cache.clear()

    def numpy(self) -> np.ndarray:
        missing = [i for i in range(self.num_layers) if i not in self._cache]
        if missing:
            raise RuntimeError(f"no activation captured for layers: {missing}")
        return np.stack([self._cache[i].numpy() for i in range(self.num_layers)], axis=1)
