from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreProtocol:
    """Fixed numerical choices used by the anonymous core implementation.

    Final fitted values such as the retained CHH count, selected rho, and CIS rank
    are data-dependent and are therefore not hard-coded here.
    """

    eps_z_scale: float = 1e-4
    eps_effect_scale: float = 0.01
    whitening_eps_fraction: float = 0.05
    gamma_fraction: float = 0.10

    bootstrap_resamples: int = 10_000
    development_bootstrap_resamples: int = 2_000
    recurrence_min: float = 0.80
    sign_consistency_min: float = 0.80
    carrier_sensitivity_max: float = 0.62

    rho_grid: tuple[float, ...] = (0.125, 0.25, 0.50, 0.75, 1.00)
    readout_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    target_fpr: float = 0.05
    max_scree_components: int = 32

    bootstrap_ci_level: float = 0.95
    ci_multiplicity: str = "none"


DEFAULT_PROTOCOL = CoreProtocol()
