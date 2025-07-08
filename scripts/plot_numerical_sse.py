from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from three_variable.simulation import (
    EtaParameters,
    SimulationConfig,
    SimulationResult,
    run_projected_simulation,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_alpha_zeta_evolution(
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
    ax1.legend(handles=[line0, line1], labels=["$\\Re(\\alpha)$", "$\\Im(\\alpha)$"])

    (line0,) = ax1.plot(result.times, result.zeta.real)
    (line1,) = ax1.twinx().plot(result.times, result.zeta.imag)
    line1.set_color("C1")
    ax1.set_title("Zeta Evolution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Zeta")
    ax1.legend(handles=[line0, line1], labels=["$\\Re(\\zeta)$", "$\\Im(\\zeta)$"])

    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_classical_evolution(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, [ax0, ax1] = plt.subplots(nrows=2)
    ax0 = cast("Axes", ax0)
    ax1 = cast("Axes", ax1)

    ax0.plot(result.times, result.x, label="$x$")
    dpdt = np.gradient(result.p, result.times)
    ax0.plot(result.times, dpdt, label="$\\frac{dp}{dt}$")
    ax0.set_title("X Evolution")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("X")
    ax0.legend()

    ax1.plot(result.times, result.p, label="$p$")
    dxdt = np.gradient(result.x, result.times)
    ax1.plot(result.times, (dxdt / 2) / result.params.eta_m, label="$\\frac{dx}{dt}$")
    ax1.set_title("P Evolution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("P")
    ax1.legend()

    fig.tight_layout()
    return fig, (ax0, ax1)


if __name__ == "__main__":
    eta_m_val = 1

    print("Running simulation")
    solution = run_projected_simulation(
        SimulationConfig(
            params=EtaParameters(
                eta_lambda=0.01,
                eta_m=eta_m_val,
                eta_omega=1,
                kbt_div_hbar=1.0,
            ),
            alpha_0=1.0 + 0.0j,
            zeta_0=-2 / 3 + 0.0j,
            times=np.linspace(0, 5, int(eta_m_val * 1000)),
        )
    )

    print("Simulation completed")
    print("Final (equilibrium) zeta:", solution.zeta[-1])

    fig, _ = plot_alpha_zeta_evolution(solution)
    fig.savefig("alpha_zeta_evolution.png", dpi=300)

    fig, _ = plot_classical_evolution(solution[-300::])
    fig.savefig("classical_evolution.png", dpi=300)

    plt.show()
