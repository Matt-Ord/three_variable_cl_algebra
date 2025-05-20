from __future__ import annotations

import numpy as np
import slate
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.util import (
    EtaParameters,
    spaced_time_basis,
)
from scipy.constants import hbar  # type: ignore library
from slate import metadata, plot
from slate_quantum import operator

from three_variable.equilibrium_squeeze import (
    evaluate_equilibrium_expect_x_squared,
)
from three_variable.physical_systems import TOWNSEND_H_RU
from three_variable.simulation import get_condition_from_params

if __name__ == "__main__":
    eta_omega, eta_lambda = 0.5, 40.0
    eta_omega, eta_lambda = (
        TOWNSEND_H_RU.eta_parameters.eta_omega,
        TOWNSEND_H_RU.eta_parameters.eta_lambda,
    )
    eta_omega, eta_lambda = 0.5, 40.0
    condition = get_condition_from_params(eta_omega, eta_lambda, mass=1)
    times = spaced_time_basis(n=100, dt=0.1 * np.pi * hbar)
    states = run_stochastic_simulation(condition, times)

    # Using the e^{ikx} operator, we can calculate the position
    # of the wavepacket.
    positions = operator.measure.all_periodic_x(states, axis=0)
    fig, ax, line = plot.array_against_basis(positions, measure="real")
    ax.set_title("Displacement of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Position (a.u.)")
    delta_x = metadata.volume.fundamental_stacked_delta_x(states.basis.metadata()[1])
    ax.set_ylim(0, delta_x[0][0])
    fig.show()
    # We have a free particle, so the wavepacket is equally likely
    # to be found at any position.
    fig, ax = plot.array_distribution(positions)
    ax.set_title("Distribution of wavepacket position")
    ax.set_xlabel("Position (a.u.)")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, delta_x[0][0])
    fig.show()

    # We can also calculate the width of the wavepacket
    # This remains almost constant over the course of the simulation.
    params = EtaParameters.from_condition(condition)
    print(params.eta_m, params.eta_omega, params.eta_lambda)
    print("", eta_omega, eta_lambda)
    params = EtaParameters(
        eta_m=params.eta_m,
        eta_omega=np.sqrt(2 / 3) * params.eta_omega,
        eta_lambda=params.eta_lambda,
    )
    theoretical = params.get_variance_x()
    theoretical = evaluate_equilibrium_expect_x_squared(
        params.eta_m,  # type: ignore sp
        params.eta_omega,  # type: ignore sp
        params.eta_lambda,  # type: ignore sp
    )

    widths = operator.measure.all_variance_x(states, axis=0)
    print(theoretical, slate.array.average(widths))
    fig, ax, line = plot.array_against_basis(widths, measure="real")
    line = ax.axhline(theoretical)  # type: ignore sp
    line.set_color("red")
    line.set_label("Theoretical width")
    ax.set_title("Width of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Width (a.u.)")
    fig.show()
    # The width of the wavepacket oscillates about the equilibrium width
    fig, ax = plot.array_distribution(widths, distribution="normal")
    line = ax.axvline(theoretical)  # type: ignore sp
    line.set_color("red")
    line.set_label("Theoretical width")
    ax.set_title("Distribution of wavepacket width")
    ax.set_xlabel("Width (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    # Similarly the uncertainty
    theoretical = np.sqrt(params.get_uncertainty() / hbar**2)
    widths = operator.measure.all_uncertainty(states, axis=0)
    fig, ax, line = plot.array_against_basis(widths, measure="real")
    line = ax.axhline(theoretical)
    line.set_color("red")
    line.set_label("Theoretical uncertainty")
    ax.set_title("Uncertainty of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Uncertainty (a.u.)")
    fig.show()
    # The uncertainty of the wavepacket oscillates about the equilibrium uncertainty
    fig, ax = plot.array_distribution(widths, distribution="normal")
    line = ax.axvline(theoretical)
    line.set_color("red")
    line.set_label("Theoretical uncertainty")
    ax.set_title("Distribution of wavepacket uncertainty")
    ax.set_xlabel("Uncertainty (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    # The simulation is not periodic in momentum space, so we can use
    # the k operator directly to calculate the momentum of the wavepacket.
    momentums = operator.measure.all_k(states, axis=0)
    fig, ax, line = plot.array_against_basis(momentums, measure="real")
    ax.set_title("Momentum of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Momentum (a.u.)")
    delta_k = metadata.volume.fundamental_stacked_delta_k(states.basis.metadata()[1])
    ax.set_ylim(-delta_k[0][0] / 2, delta_k[0][0] / 2)
    fig.show()
    # The distribution of momentum of the wavepacket is centered at zero,
    fig, ax = plot.array_distribution(momentums, distribution="normal")
    ax.set_title("Distribution of wavepacket momentum")
    ax.set_xlabel("Momentum (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    plot.wait_for_close()
