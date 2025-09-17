from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import MaxNLocator
from scipy.constants import Boltzmann  # type: ignore[import-untyped]
from scipy.constants import hbar as hbar_value  # type: ignore[import-untyped]
from slate_core.util import timed

from three_variable.coherent_states import (
    expectation_from_formula,
)
from three_variable.equilibrium_squeeze import (
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import get_harmonic_term, get_kinetic_term
from three_variable.simulation import (
    EtaParameters,
    SimulationConfig,
    SimulationResult,
    evaluate_equilibrium_squeeze_ratio,
    run_projected_simulation,
)
from three_variable.simulation.physical_systems import ELENA_NA_CU
from three_variable.symbols import (
    KBT,
    alpha,
    dimensionless_from_full,
    eta_lambda,
    eta_m,
    eta_omega,
    formula_from_expr,
    hbar,
    zeta,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from three_variable.simulation._projected_equations import SimulationResult


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


def plot_classical_eom_comparison(
    result: SimulationResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, [ax0, ax1] = plt.subplots(nrows=2)
    ax0 = cast("Axes", ax0)
    ax1 = cast("Axes", ax1)
    kb_t_value = result.params.kbt_div_hbar * hbar_value
    stiffness = -0.5 * result.params.eta_m * kb_t_value / (result.params.eta_omega**2)
    ax0.plot(result.times, stiffness * result.x, label=r"$-x m \omega^2$")
    dpdt = np.gradient(result.p, result.times)
    ax0.plot(result.times, dpdt, label="$\\frac{dp}{dt}$")
    ax0.set_title("X Evolution")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("X")
    ax0.legend()

    ax1.plot(result.times, result.p, label="$p$")
    dxdt = np.gradient(result.x, result.times)
    m = result.params.eta_m * hbar_value**2 / (2 * kb_t_value)
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
    kb_t_value = result.params.kbt_div_hbar * hbar_value
    stiffness = -0.5 * result.params.eta_m * kb_t_value / (result.params.eta_omega**2)
    ax0.plot(result.times, stiffness * result.x, label="$x$")
    dpdt = np.gradient(result.p, result.times)
    ax0.plot(result.times, dpdt, label="$\\frac{dp}{dt}$")
    ax0.set_title("X Evolution")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("X")
    ax0.legend()

    ax1.plot(result.times, result.p, label="$p$")
    m = result.params.eta_m * hbar_value**2 / (2 * kb_t_value)
    ax1.plot(
        result.times, result.x_derivative_equilibrium * m, label="$\\frac{dx}{dt} m$"
    )
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
    ax0_line = ax0.axhline(0)
    ax0_line.set_alpha(0.5)
    ax0_line.set_linestyle("--")
    ax0_line.set_color("black")
    (line0,) = ax0.plot(times, result.x, label="$x$")
    line0.set_color("C0")

    ax1_line = ax1.axhline(0)
    ax1_line.set_alpha(0.5)
    ax1_line.set_linestyle("--")
    ax1_line.set_color("black")
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

    expect_zeta_r0 = squeeze_ratio_from_zeta_expr(zeta).subs(eta_m, 1)
    expect_zeta_fn = sp.lambdify((squeeze_ratio), expect_zeta_r0, modules="numpy")
    zeta_vals = np.asarray(expect_zeta_fn(result.squeeze_ratio))

    (line0,) = ax0.plot(result.times, np.abs(zeta_vals))
    ax1 = ax0.twinx()
    (line1,) = ax1.plot(result.times, np.unwrap(np.angle(zeta_vals)))
    line1.set_color("C1")
    ax0.set_title("Squeezing Evolution")
    ax0.set_xlabel("Time /au")
    ax0.set_ylabel("Squeezing $r$")
    ax1.set_ylabel(r"Squeezing $\theta$")
    ax1.legend(handles=[line0, line1], labels=["$r$", r"$\theta$"])

    fig.tight_layout()
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, (ax0, ax1)


@timed
def _get_energy_from_simulation(
    result: SimulationResult,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Calculate the energy distribution of the system."""
    energy = get_kinetic_term() + get_harmonic_term()
    energy = formula_from_expr(energy)
    energy = expectation_from_formula(energy)
    energy = squeeze_ratio_from_zeta_expr(energy)

    energy = dimensionless_from_full(energy)
    energy = energy.subs(
        {
            eta_lambda: result.params.eta_lambda,
            eta_omega: result.params.eta_omega,
            eta_m: result.params.eta_m,
            hbar: 1,
            KBT: result.params.kbt_div_hbar,
        }
    )

    energy_fn = sp.lambdify(
        (alpha, squeeze_ratio),
        energy,
        modules="numpy",
    )
    return np.real(energy_fn(result.alpha, result.squeeze_ratio))  # type: ignore[misc]


def _get_binned_energy(
    energy: np.ndarray, *, n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    energy_hist, bin_edges = np.histogram(energy, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_centers = bin_centers[energy_hist > 0]
    energy_hist = energy_hist[energy_hist > 0]
    return bin_centers, energy_hist


def _fit_to_kbt_line(
    energy: np.ndarray,
    density: np.ndarray,
) -> tuple[float, float]:
    slope, intercept = np.polyfit(energy, np.log(density), 1)
    return slope, intercept


def plot_energy_distribution(
    result: SimulationResult,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()

    energy = _get_energy_from_simulation(result)
    binned_energy, density = _get_binned_energy(energy)

    ax.plot(binned_energy, density, label="Energy Distribution")
    ax.set_xlabel("Energy / hbar")
    ax.set_ylabel("Log Probability Density")
    ax.set_title("Energy Distribution of the System")

    slope, intercept = _fit_to_kbt_line(binned_energy, density)
    fitted = np.exp(slope * binned_energy + intercept)
    (line,) = ax.plot(binned_energy, fitted, label="Fit")
    line.set_linestyle("--")
    expected = np.exp((-1 / result.params.kbt_div_hbar) * binned_energy + intercept)
    (line,) = ax.plot(binned_energy, expected, label="Expected from KBT")
    line.set_linestyle("--")
    ax.legend()
    ax.set_yscale("log")
    return fig, ax


if __name__ == "__main__":
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    eta_lambda_val = ELENA_NA_CU.eta_parameters.eta_lambda
    eta_omega_val = ELENA_NA_CU.eta_parameters.eta_omega
    time_scale = hbar_value / 300 * Boltzmann
    print("Estimating initial r0")
    equilibrium_ratio = evaluate_equilibrium_squeeze_ratio(
        eta_lambda_val=eta_lambda_val,
        eta_omega_val=eta_omega_val,
    )
    print("Running simulation")
    alpha_config = SimulationConfig(
        params=EtaParameters(
            eta_lambda=eta_lambda_val,
            eta_m=eta_m_val,
            eta_omega=eta_omega_val,
            kbt_div_hbar=1 / time_scale,
        ),
        alpha_0=0.0 + 0.0j,
        r_0=equilibrium_ratio,
        times=np.linspace(0, 20, 1000) * time_scale,
    )
    solution = run_projected_simulation(alpha_config)

    print("Simulation completed")
    print("Final (equilibrium) r:", solution.squeeze_ratio[-1])
    print("Error in r:", equilibrium_ratio - solution.squeeze_ratio[-1])

    fig, _ = plot_alpha_r_evolution(solution)
    fig.savefig("alpha_r_evolution.png", dpi=300)

    fig, _ = plot_classical_eom_comparison(solution)
    fig.savefig("classical_eom_comparison.png", dpi=300)

    fig, _ = plot_classical_evolution(solution)
    fig.savefig("classical_evolution.png", dpi=300)

    solution = run_projected_simulation(
        dataclasses.replace(
            alpha_config,
            times=np.linspace(0, 20000, 100000) * time_scale,
        )
    )
    fig, _ = plot_energy_distribution(solution)
    fig.savefig("energy_distribution.png", dpi=300)

    solution = run_projected_simulation(
        SimulationConfig(
            params=EtaParameters(
                eta_lambda=eta_lambda_val,
                eta_m=eta_m_val,
                eta_omega=eta_omega_val,
                kbt_div_hbar=1 / time_scale,
            ),
            alpha_0=0.0 + 0.0j,
            r_0=1.5 * equilibrium_ratio * np.exp(1j * 0.1),
            times=np.linspace(0, 50, 10000) * time_scale,
        )
    )
    fig, _ = plot_r_theta_evolution(solution)
    fig.savefig("r_theta_evolution.png", dpi=600)

    plt.show()
