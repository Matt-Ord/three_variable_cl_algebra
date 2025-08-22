from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adsorbate_simulation.system import (
        CaldeiraLeggettSimulationConfig,
        HarmonicPotential,
        SimulationCondition,
        System,
    )

try:
    from three_variable.simulation.condition import (
        get_condition_from_params,
    )
except ImportError:

    def get_condition_from_params(  # noqa: PLR0913
        eta_omega: float,  # noqa: ARG001
        eta_lambda: float,  # noqa: ARG001
        *,
        mass: float = 0,  # noqa: ARG001
        temperature: float = 0,  # noqa: ARG001
        minimum_occupation: float = 1e-6,  # noqa: ARG001
        truncate: bool = True,  # noqa: ARG001
    ) -> SimulationCondition[
        System[HarmonicPotential], CaldeiraLeggettSimulationConfig
    ]:
        msg = (
            "The 'three_variable.simulation.condition' module is not available. "
            "Please add 'three_variable[simulate]' to your requirements to install the necessary dependencies."
        )
        raise ImportError(msg)


from three_variable.simulation._projected_equations import (
    KBT_value,
    SimulationConfig,
    SimulationResult,
    evaluate_equilibrium_squeeze_ratio,
    explicit_from_dimensionless,
    hbar_value,
    run_projected_simulation,
)
from three_variable.simulation.physical_systems import (
    ELENA_CU_LATTICE_PARAMETER,
    ELENA_LI_CU,
    ELENA_NA_CU,
    ELENA_NA_CU_BALLISTIC,
    HYDROGEN_MASS,
    LI_MASS,
    NA_MASS,
    TOWNSEND_H_RU,
    BaseParameters,
    ElenaParameters,
    EtaParameters,
)

__all__ = [
    "ELENA_CU_LATTICE_PARAMETER",
    "ELENA_LI_CU",
    "ELENA_NA_CU",
    "ELENA_NA_CU_BALLISTIC",
    "HYDROGEN_MASS",
    "LI_MASS",
    "NA_MASS",
    "TOWNSEND_H_RU",
    "BaseParameters",
    "ElenaParameters",
    "EtaParameters",
    "KBT_value",
    "SimulationConfig",
    "SimulationResult",
    "evaluate_equilibrium_squeeze_ratio",
    "explicit_from_dimensionless",
    "get_condition_from_params",
    "hbar_value",
    "run_projected_simulation",
]
