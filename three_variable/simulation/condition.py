from __future__ import annotations

import numpy as np
from adsorbate_simulation.constants import DIMENSIONLESS_1D_SYSTEM
from adsorbate_simulation.system import (
    CaldeiraLeggettEnvironment,
    CaldeiraLeggettSimulationConfig,
    HarmonicCoherentInitialState,
    HarmonicPotential,
    PositionSimulationBasis,
    SimulationCondition,
    System,
)
from scipy.constants import Boltzmann, hbar  # type: ignore lib


def _get_condition_energy(
    temperature: float = 10 / Boltzmann, minimum_occupation: float = 1e-6
) -> float:
    """Get the highest energy state that we include in the simulation."""
    return -Boltzmann * temperature * np.log(minimum_occupation)


def _get_width_at_energy(energy: float, frequency: float) -> float:
    """V(x) = 1/2 omega^2 X ^2."""
    return 2 * np.sqrt(2 * energy) / frequency


def get_condition_from_params(  # noqa: PLR0913
    eta_omega: float,
    eta_lambda: float,
    *,
    mass: float = DIMENSIONLESS_1D_SYSTEM.mass,
    temperature: float = 10 / Boltzmann,
    minimum_occupation: float = 1e-6,
    truncate: bool = True,
) -> SimulationCondition[System[HarmonicPotential], CaldeiraLeggettSimulationConfig]:
    """Get a dimensionless system, scaled such that temperature = 10 / Boltzmann.

    minimum_occupation gives the minimum classical probability of occupation that states have
    of being occupied. for example for minimum_occupation = 1e-3 we include only momentum states
    with e^(-beta E_k) >= minimum_occupation and position states with e^(-beta V(x)) >= minimum_occupation
    """
    omega = Boltzmann * temperature / (hbar * eta_omega)
    frequency = np.sqrt(mass) * omega
    gamma = Boltzmann * temperature / (hbar * eta_lambda)

    initial_state = HarmonicCoherentInitialState()
    width = initial_state.get_harmonic_width(frequency, mass)
    # We choose the cell length so that the particle does not reach the boundary.
    # This has an energy V(x) = 1/2 f^2 (delta_x * (1/2) * (2/3))^2 since we
    # truncate the basis to include only 2/3 of the position states.
    max_energy = _get_condition_energy(temperature, minimum_occupation)
    # Note, we also need to add space for the whole wavepacket to fit.
    # Not just the center! We assume that the wavepacket has roughly
    # the same width as the initial state.
    delta_x = _get_width_at_energy(max_energy, frequency) + 6 * width
    delta_x = (3 / 2) * delta_x if truncate else delta_x
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
        CaldeiraLeggettSimulationConfig(
            simulation_basis=PositionSimulationBasis(
                shape=(1,),
                resolution=(n_k,),
                offset=((n_k - n_states) // 2,),
                truncation=(n_states,),
            )
            if truncate
            else PositionSimulationBasis(shape=(1,), resolution=(n_k,)),
            environment=CaldeiraLeggettEnvironment.from_gamma(gamma, system.mass),
            temperature=10 / Boltzmann,
            target_delta=0.2e-3,
            initial_state=initial_state,
        ),
    )
