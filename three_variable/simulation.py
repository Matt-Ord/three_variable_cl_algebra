from __future__ import annotations

import numpy as np
from adsorbate_simulation.constants import DIMENSIONLESS_1D_SYSTEM
from adsorbate_simulation.system import (
    CaldeiraLeggettEnvironment,
    HarmonicCoherentInitialState,
    HarmonicPotential,
    IsotropicSimulationConfig,
    PositionSimulationBasis,
    SimulationCondition,
    System,
)
from scipy.constants import Boltzmann, hbar  # type: ignore lib


def get_condition_from_params(
    eta_omega: float,
    eta_lambda: float,
    *,
    mass: float = DIMENSIONLESS_1D_SYSTEM.mass,
    temperature: float = 10 / Boltzmann,
    minumum_occupation: float = 1e-6,
) -> SimulationCondition[System[HarmonicPotential], IsotropicSimulationConfig]:
    """Get a dimensionless system, scaled such that temperature = 10 / Boltzmann.

    minumum_occupation gives the minimum classical probability of occupation that states have
    of being occupied. for example for minumum_occupation = 1e-3 we include only momentum states
    with e^(-beta E_k) >= minumum_occupation and position states with e^(-beta V(x)) >= minumum_occupation
    """
    omega = Boltzmann * temperature / (hbar * eta_omega)
    frequency = np.sqrt(mass) * omega
    gamma = Boltzmann * temperature / (hbar * eta_lambda)

    initial_state = HarmonicCoherentInitialState()
    width = initial_state.get_harmonic_width(frequency, mass)
    # We choose the cell length so that the particle does not reach the boundary.
    # This has an energy V(x) = 1/2 f^2 (delta_x * (1/2) * (2/3))^2 since we
    # truncate the basis to include only 2/3 of the position states.
    max_energy = -Boltzmann * temperature * np.log(minumum_occupation)
    # Note, we also need to add space for the whole wavepacket to fit.
    # Not just the center! We assume that the wavepacket has roughly
    # the same width as the initial state.
    delta_x = (3 / frequency) * np.sqrt(2 * max_energy) + 6 * width
    max_k = (1 / hbar) * np.sqrt(2 * mass * max_energy) + 4 * (2 * np.pi / width)
    d_k = (2 * np.pi) / delta_x
    # We need both positive and negative k states
    n_k = int(2 * np.ceil(max_k / d_k))
    # truncate the basis to include only 2/3 of the position states.
    n_states = int(np.ceil(2 * n_k / 3))

    system = System(
        mass=mass,
        potential=HarmonicPotential(frequency=frequency),
        cell=DIMENSIONLESS_1D_SYSTEM.cell.with_lengths((delta_x,)),
    )

    return SimulationCondition(
        system,
        IsotropicSimulationConfig(
            simulation_basis=PositionSimulationBasis(
                shape=(1,),
                resolution=(n_k,),
                offset=((n_k - n_states) // 2,),
                truncation=(n_states,),
            ),
            environment=CaldeiraLeggettEnvironment.from_gamma(gamma, system.mass),
            temperature=10 / Boltzmann,
            target_delta=0.2e-3,
            initial_state=initial_state,
        ),
    )
