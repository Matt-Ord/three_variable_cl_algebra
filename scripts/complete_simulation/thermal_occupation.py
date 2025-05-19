from __future__ import annotations

import numpy as np
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.util import (
    get_eigenvalue_occupation_hermitian,
    spaced_time_basis,
)
from matplotlib.scale import SymmetricalLogScale
from scipy.constants import hbar  # type: ignore libary
from slate import array, linalg, plot
from slate_quantum import state

from three_variable.physical_systems import TOWNSEND_H_RU
from three_variable.simulation import get_condition_from_params

if __name__ == "__main__":
    eta_omega, eta_lambda = (
        TOWNSEND_H_RU.eta_parameters.eta_omega,
        TOWNSEND_H_RU.eta_parameters.eta_lambda,
    )
    condition = get_condition_from_params(eta_omega, eta_lambda)
    eta_omega, eta_lambda = 0.5, 40.0
    condition = get_condition_from_params(eta_omega, eta_lambda, mass=hbar**2)
    print(condition)
    # We simulate the system using the stochastic Schrodinger equation.
    # We find a localized stochastic evolution of the wavepacket.
    times = spaced_time_basis(n=10, dt=0.1 * np.pi * hbar)
    states = run_stochastic_simulation.call_cached(condition, times)
    states = state.normalize_all(states)

    condition = condition.with_temperature(condition.config.temperature * (3 / 2))
    diagonal_hamiltonian = linalg.into_diagonal_hermitian(condition.hamiltonian)
    target_occupation = get_eigenvalue_occupation_hermitian(
        diagonal_hamiltonian, condition.config.temperature
    )

    states = states.with_state_basis(diagonal_hamiltonian.basis.inner[0])

    average_occupation, std_occupation = state.get_average_occupations(states)
    average_occupation = array.cast_basis(average_occupation, target_occupation.basis)
    std_occupation = array.cast_basis(std_occupation, target_occupation.basis)
    # We see that the true occupation of the states is close to the
    # expected thermal occupation.
    fig, ax = plot.get_figure()
    _, _, line = plot.array_against_basis(target_occupation, ax=ax)
    line.set_label("Target Occupation")
    _, _, line = plot.array_against_basis(
        average_occupation, y_error=std_occupation, ax=ax
    )
    line.set_marker("x")
    line.set_label("Average Occupation")
    ax.set_title("True occupation of the states")
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    ax.set_xlim(0, 300)
    ax.legend()
    ax.set_yscale(SymmetricalLogScale(None, linthresh=1e-5))
    line.set_marker("x")
    fig.show()

    plot.wait_for_close()
