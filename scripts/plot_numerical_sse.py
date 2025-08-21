from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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


def plot_classical_eom_comparison(
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


def plot_classical_evolution(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, ax0 = plt.subplots(figsize=(5, 3))
    ax1 = ax0.twinx()

    times = result.times - result.times[0]
    (line0,) = ax0.plot(times, result.x, label="$x$")
    line0.set_color("C0")

    (line1,) = ax1.plot(times, result.p, label="$p$")
    line1.set_color("C1")

    # Offset the two plots, so they don't overlap
    start, end = ax0.get_ylim()
    ax0.set_ylim(start - 0.5 * (end - start), end)
    start, end = ax1.get_ylim()
    ax1.set_ylim(start, end + 0.5 * (end - start))

    ax0.set_xlabel("Time /au")
    ax0.set_ylabel("X /au")
    ax1.set_ylabel("P /au")
    ax0.set_title("Evolution of X and P")
    ax1.legend(handles=[line0, line1], labels=["$x$", "$p$"])

    fig.tight_layout()
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, (ax0, ax1)


def plot_r_theta_evolution(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, ax0 = plt.subplots(figsize=(5, 3))

    (line0,) = ax0.plot(result.times, np.abs(result.zeta))
    ax1 = ax0.twinx()
    (line1,) = ax1.plot(result.times, np.unwrap(np.angle(result.zeta)))
    line1.set_color("C1")
    ax0.set_title("Squeezing Evolution")
    ax0.set_xlabel("Time /au")
    ax0.set_ylabel("Squeezing $r$")
    ax1.set_ylabel(r"Squeezing $\theta$")
    ax0.set_ylim(0, None)
    ax1.set_ylim(0, None)
    ax1.legend(handles=[line0, line1], labels=["$r$", r"$\theta$"])

    fig.tight_layout()
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=4))
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
            alpha_0=0.0 + 0.0j,
            zeta_0=-0.002 + 0.0j,
            times=np.linspace(0, 5, int(eta_m_val * 1000)),
        )
    )

    print("Simulation completed")
    print("Final (equilibrium) zeta:", solution.zeta[-1])

    fig, _ = plot_alpha_zeta_evolution(solution)
    fig.savefig("alpha_zeta_evolution.png", dpi=300)

    fig, _ = plot_classical_eom_comparison(solution[-300::])
    fig.savefig("classical_eom_comparison.png", dpi=300)

    fig, _ = plot_classical_evolution(solution[-300::])
    fig.savefig("classical_evolution.png", dpi=300)

    solution = run_projected_simulation(
        SimulationConfig(
            params=EtaParameters(
                eta_lambda=0.01,
                eta_m=eta_m_val,
                eta_omega=1,
                kbt_div_hbar=1.0,
            ),
            alpha_0=0.0 + 0.0j,
            zeta_0=0.005 * np.exp(1j * 0.1),
            times=np.linspace(0, 0.03, int(eta_m_val * 1000)),
        )
    )
    fig, _ = plot_r_theta_evolution(solution)
    fig.savefig("r_theta_evolution.png", dpi=600)

    plt.show()
