from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping

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
    """Resolve language self-attention ``o_proj`` modules in layer order."""
    num_layers, _, _ = attention_shape(model)
    primary = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn\.o_proj$")
    fallback = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")
    found: dict[int, AttentionOutputProjection] = {}

    for name, module in model.named_modules():
        match = primary.search(name)
        if match:
            layer = int(match.group(1))
            found[layer] = AttentionOutputProjection(layer, name, module)

    if len(found) != num_layers:
        for name, module in model.named_modules():
            if "visual" in name or "vision" in name:
                continue
            match = fallback.search(name)
            if match:
                layer = int(match.group(1))
                found.setdefault(layer, AttentionOutputProjection(layer, name, module))

    missing = [layer for layer in range(num_layers) if layer not in found]
    if missing:
        raise RuntimeError(f"could not resolve language attention output projections: missing {missing}")
    return [found[layer] for layer in range(num_layers)]


class HeadOutputRecorder:
    """Record final-token per-head outputs immediately before ``W_O``."""

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
                raise RuntimeError(f"attention width {x.shape[-1]} != {expected}")
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
        missing = [layer for layer in range(self.num_layers) if layer not in self._cache]
        if missing:
            raise RuntimeError(f"no activation captured for layers: {missing}")
        return np.stack([self._cache[layer].numpy() for layer in range(self.num_layers)], axis=1)


class HeadOutputPatcher:
    """Optional pre-``W_O`` patcher for matched residual interventions.

    ``patches`` maps a layer to ``(head_indices, vectors)`` where vectors have
    shape ``[K, head_dim]``. In ``mode="add"`` the vectors are added at the
    fixed conditioning token and a negative ``alpha`` subtracts an interaction
    residual. In ``mode="replace"`` the vectors replace the selected head
    states, which supports natural-donor controls without synthesizing an
    off-manifold arithmetic residual.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patches: Mapping[int, tuple[np.ndarray, np.ndarray]],
        alpha: float = -1.0,
        token_index: int = -1,
        projections: Iterable[AttentionOutputProjection] | None = None,
        mode: str = "add",
    ) -> None:
        self.model = model
        self.patches = patches
        self.alpha = float(alpha)
        self.token_index = int(token_index)
        self.projections = list(projections or discover_attention_output_projections(model))
        if mode not in {"add", "replace"}:
            raise ValueError("mode must be 'add' or 'replace'")
        self.mode = mode
        self.num_layers, self.num_heads, self.head_dim = attention_shape(model)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _pre_hook(self, layer: int):
        def hook(_module, args):
            if layer not in self.patches:
                return None
            x = args[0]
            head_idx, vectors = self.patches[layer]
            head_idx = np.asarray(head_idx, dtype=int)
            vectors = np.asarray(vectors)
            if vectors.shape != (len(head_idx), self.head_dim):
                raise ValueError(f"bad patch shape for layer {layer}: {vectors.shape}")
            y = x.clone()
            view = y.reshape(y.shape[0], y.shape[1], self.num_heads, self.head_dim)
            patch = torch.as_tensor(vectors, dtype=view.dtype, device=view.device)
            if self.mode == "add":
                view[:, self.token_index, head_idx, :] += self.alpha * patch[None, :, :]
            else:
                view[:, self.token_index, head_idx, :] = patch[None, :, :]
            return (view.reshape_as(y),) + tuple(args[1:])
        return hook

    def __enter__(self):
        for item in self.projections:
            self._handles.append(item.module.register_forward_pre_hook(self._pre_hook(item.layer)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
