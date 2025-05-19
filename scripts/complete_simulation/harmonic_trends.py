from __future__ import annotations

from typing import TYPE_CHECKING

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_SYSTEM
from adsorbate_simulation.system._condition import SimulationCondition
from adsorbate_simulation.system._config import IsotropicSimulationConfig

if TYPE_CHECKING:
    from adsorbate_simulation.system._potential import HarmonicPotential
    from adsorbate_simulation.system._system import System
    from adsorbate_simulation.util import EtaParameters


def get_condition_from_params(
    params: EtaParameters,
) -> SimulationCondition[System[HarmonicPotential], IsotropicSimulationConfig]:
    return SimulationCondition(
        DIMENSIONLESS_1D_SYSTEM,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(
                shape=(2,), resolution=(55,), truncation=(2 * 45,)
            ),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=eta),
            temperature=10 / Boltzmann,
            target_delta=1e-3,
        ),
    )
