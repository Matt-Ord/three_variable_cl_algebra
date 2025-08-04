from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from three_variable.simulation import (
    EtaParameters,
    KBT_value,
    SimulationConfig,
    SimulationResult,
    estimate_r0,
    hbar_value,
    run_projected_simulation,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_alpha_r_evolution(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, [ax0, ax1] = plt.subplots(nrows=2)
    ax0 = cast("Axes", ax0)
    ax1 = cast("Axes", ax1)

    (line0,) = ax0.plot(result.times, result.alpha.real)
    (line1,) = ax0.twinx().plot(result.times, result.alpha.imag)
    line1.set_color("C1")
    ax0.set_title("Alpha Evolution")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Alpha")
    ax0.legend(handles=[line0, line1], labels=["$\\Re(\\alpha)$", "$\\Im(\\alpha)$"])

    (line0,) = ax1.plot(result.times, result.squeeze_ratio.real)
    (line1,) = ax1.twinx().plot(result.times, result.squeeze_ratio.imag)
    line1.set_color("C1")
    ax1.set_title("r0 Evolution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("r0")
    ax1.legend(handles=[line0, line1], labels=["$\\Re(R0)$", "$\\Im(R0)$"])

    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_classical_evolution(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, [ax0, ax1] = plt.subplots(nrows=2)
    ax0 = cast("Axes", ax0)
    ax1 = cast("Axes", ax1)
    stiffness = -0.5 * result.params.eta_m * KBT_value / (result.params.eta_omega**2)
    ax0.plot(result.times, stiffness * result.x, label=r"$-x m \omega^2$")
    dpdt = np.gradient(result.p, result.times)
    ax0.plot(result.times, dpdt, label="$\\frac{dp}{dt}$")
    ax0.set_title("X Evolution")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("X")
    ax0.legend()

    ax1.plot(result.times, result.p, label="$p$")
    dxdt = np.gradient(result.x, result.times)
    m = result.params.eta_m * hbar_value**2 / (2 * KBT_value)
    ax1.plot(result.times, dxdt * m, label="$\\frac{dx}{dt} m$")
    ax1.set_title("P Evolution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("P")
    ax1.legend()

    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_classical_evolution_formula(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, [ax0, ax1] = plt.subplots(nrows=2)
    ax0 = cast("Axes", ax0)
    ax1 = cast("Axes", ax1)
    stiffness = -0.5 * result.params.eta_m * KBT_value / (result.params.eta_omega**2)
    ax0.plot(result.times, stiffness * result.x, label="$x$")
    dpdt = np.gradient(result.p, result.times)
    ax0.plot(result.times, dpdt, label="$\\frac{dp}{dt}$")
    ax0.set_title("X Evolution")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("X")
    ax0.legend()

    ax1.plot(result.times, result.p, label="$p$")
    m = result.params.eta_m * hbar_value**2 / (2 * KBT_value)
    ax1.plot(
        result.times, result.x_derivative_equilibrium * m, label="$\\frac{dx}{dt} m$"
    )
    ax1.set_title("P Evolution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("P")
    ax1.legend()

    fig.tight_layout()
    return fig, (ax0, ax1)


if __name__ == "__main__":
    eta_m_val = 1e22
    eta_lamda_val = 60
    eta_omega_val = 1
    time_scale = hbar_value / KBT_value
    print("Estimating initial r0")
    r0_estimate = estimate_r0(
        eta_lambda_val=eta_lamda_val,
        eta_omega_val=eta_omega_val,
    )
    print("Running simulation")
    solution = run_projected_simulation(
        SimulationConfig(
            params=EtaParameters(
                eta_lambda=eta_lamda_val,
                eta_m=eta_m_val,
                eta_omega=eta_omega_val,
                kbt_div_hbar=1,
            ),
            alpha_0=0.000001 + 0.0j,
            r_0=r0_estimate,
            times=np.linspace(0, 1000, 50000) * time_scale,
        )
    )

    print("Simulation completed")
    print("Final (equilibrium) r:", solution.squeeze_ratio[-1])

    fig, _ = plot_alpha_r_evolution(solution)
    fig.savefig("alpha_r_evolution.png", dpi=300)

    fig, _ = plot_classical_evolution(solution[-1000::])
    fig.savefig("classical_evolution.png", dpi=300)

    plt.show()
