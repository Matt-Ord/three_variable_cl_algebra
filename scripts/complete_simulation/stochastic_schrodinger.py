from __future__ import annotations

import numpy as np
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.util import spaced_time_basis
from scipy.constants import hbar  # type: ignore libary
from slate_core import basis, plot
from slate_core.plot import (
    animate_data_over_list_1d_k,
    animate_data_over_list_1d_x,
)
from slate_quantum import dynamics, state

from three_variable.physical_systems import TOWNSEND_H_RU
from three_variable.simulation import get_condition_from_params

if __name__ == "__main__":
    eta_omega, eta_lambda = (
        TOWNSEND_H_RU.eta_parameters.eta_omega,
        TOWNSEND_H_RU.eta_parameters.eta_lambda,
    )
    eta_omega, eta_lambda = 0.5, 40.0
    condition = get_condition_from_params(eta_omega, eta_lambda, mass=1)
    # We simulate the system using the stochastic Schrodinger equation.
    # We find a localized stochastic evolution of the wavepacket.
    times = spaced_time_basis(n=100, dt=0.1 * np.pi * hbar)
    states = run_stochastic_simulation.call_cached(condition, times)
    states = dynamics.select_realization(states)

    # We start the system in a gaussian state, centered at the origin.
    fig, ax, _ = plot.array_against_axes_1d(states[0, :], measure="abs")
    line = ax.axvline(1 / 6 * states.basis.metadata().children[1].children[0].delta)
    line.set_color("black")
    line = ax.axvline(5 / 6 * states.basis.metadata().children[1].children[0].delta)
    line.set_color("black")
    ax.set_xlim(0, states.basis.metadata().children[1].children[0].delta)
    ax.set_title("Initial State - A Gaussian Wavepacket Centered at the Origin")
    fig.show()

    fig, ax, anim0 = animate_data_over_list_1d_x(states, measure="abs")
    ax.set_title("Stochastic Evolution of the Wavepacket")
    ax.set_xlabel("Position (a.u.)")
    ax.set_ylabel("Probability")
    line = ax.axvline(1 / 6 * states.basis.metadata().children[1].children[0].delta)
    line.set_color("black")
    line = ax.axvline(5 / 6 * states.basis.metadata().children[1].children[0].delta)
    line.set_color("black")
    fig.show()
    fig, ax, anim1 = animate_data_over_list_1d_k(states, measure="abs")
    ax.set_title("Stochastic Evolution of the Wavepacket in k-space")
    ax.set_xlabel("Momentum (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    # Check that the states are normalized - this is an easy way to check that
    # we have a small enough time step.
    normalization = state.all_inner_product(states, states)
    fig, ax, line = plot.array_against_basis(normalization, measure="real")
    ax.set_title("Normalization of the states against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Normalization")
    ylim = ax.get_ylim()
    delta = max(1 - ylim[0], ylim[1] - 1)
    ax.set_ylim(1 - delta, 1 + delta)
    fig.show()

    plot.wait_for_close()

    basis_list = basis.as_index(basis.as_tuple(states.basis).children[0])
    for i in basis_list.points:
        s = states[i.item(), :]
        assert np.isclose(1, normalization.as_array(), atol=1e-2)
