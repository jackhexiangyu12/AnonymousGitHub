"""Minimal CAMP-IE utilities for conditioner head extraction and factorial head analysis."""

from .modeling import QwenConditioner, load_qwen_conditioner
from .hooks import HeadOutputRecorder, discover_attention_output_projections
from .head_analysis import (
    HeadDirection,
    FactorialHeadEffects,
    fit_harmfulness_direction,
    project_head_scores,
    factorial_head_effects,
    rank_positive_heads,
)

__all__ = [
    "QwenConditioner",
    "load_qwen_conditioner",
    "HeadOutputRecorder",
    "discover_attention_output_projections",
    "HeadDirection",
    "FactorialHeadEffects",
    "fit_harmfulness_direction",
    "project_head_scores",
    "factorial_head_effects",
    "rank_positive_heads",
]
