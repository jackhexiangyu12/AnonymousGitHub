"""Core CAMP-IE implementation used for anonymous review."""

from .cis import CISState, load_cis_state
from .head_analysis import (
    FactorialHeadEffects,
    HeadDirection,
    StableHead,
    factorial_head_effects,
    fit_layer_harmfulness_direction,
    project_head_scores,
    select_stable_heads,
)
from .protocol import CoreProtocol, DEFAULT_PROTOCOL

__all__ = [
    "CISState",
    "CoreProtocol",
    "DEFAULT_PROTOCOL",
    "FactorialHeadEffects",
    "HeadDirection",
    "StableHead",
    "factorial_head_effects",
    "fit_layer_harmfulness_direction",
    "load_cis_state",
    "project_head_scores",
    "select_stable_heads",
]
