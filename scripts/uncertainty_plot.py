from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from matplotlib.colors import LogNorm
from slate_core import plot
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    uncertainty_squared,
)
from three_variable.equilibrium_squeeze import (
    evaluate_equilibrium_expect_x_squared,
    evaluate_equilibrium_uncertainty,
    get_equilibrium_squeeze_ratio,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.simulation import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU
from three_variable.symbols import (
    eta_lambda as _eta_lambda_symbol,
)
from three_variable.symbols import eta_m as _eta_m_symbol
from three_variable.symbols import eta_omega

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]
print("Omega      | m          | lambda        ")
print(f"{TOWNSEND_H_RU.omega:0.3e}", TOWNSEND_H_RU.m, f"{TOWNSEND_H_RU.lambda_:0.3e}")
print(f"{ELENA_LI_CU.omega:0.3e}", ELENA_LI_CU.m, f"{ELENA_LI_CU.lambda_:0.3e}")
print(f"{ELENA_NA_CU.omega:0.3e}", ELENA_NA_CU.m, f"{ELENA_NA_CU.lambda_:0.3e}")


def plot_uncertainty_against_m_w() -> None:
    eta_m = np.linspace(0.0001, 2, 1000)
    eta_omega = np.linspace(0.0001, 10, 1000)
    eta_m_grid, eta_omega_grid = np.meshgrid(eta_m, eta_omega)
    eta_lambda = np.ones_like(eta_m_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m_grid, eta_omega_grid, eta_lambda
    )

    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_m_grid, eta_omega_grid, uncertainty)
    mesh.set_norm(LogNorm())
    mesh.set_clim(1 / 16, None)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty against $\eta_m$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\omega$")
    fig.show()


def plot_uncertainty_against_lambda_w() -> None:
    eta_lambda = np.linspace(0.001, 20, 1000)
    eta_omega = np.linspace(0.001, 20, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1 * np.ones_like(eta_lambda_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m, eta_omega_grid, eta_lambda_grid
    )
    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, 2 * np.sqrt(uncertainty) - 1)
    mesh.set_clim(0, None)

    # Plot the classical threshold
    def get_threshold_from_omega(
        eta_omega: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return (1 - 4 * eta_omega) ** 2 / 2

    def get_threshold_from_omega_lower(
        eta_omega: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return (2 - np.sqrt(2)) / np.abs(
            (8 - 6 * np.sqrt(2)) + ((2 * np.sqrt(2)) / (4 * eta_omega) ** 2)
        )

    old_lim = ax.get_xlim()
    (line,) = ax.plot(get_threshold_from_omega(eta_omega), eta_omega)
    line.set_label("Classical threshold")
    line.set_color("C5")
    line.set_linestyle("--")
    (line,) = ax.plot(get_threshold_from_omega_lower(eta_omega), eta_omega)
    line.set_color("C6")
    line.set_linestyle("--")
    ax.set_xlim(old_lim)

    for name, parameters, c in PHYSICAL_PARAMS:
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)  # type: ignore unknown
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_title(
        r"Plot of quantum uncertainty against $\eta_\lambda$ and $\eta_\omega$"
    )
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()


def plot_uncertainty_against_lambda_w_logspace() -> None:
    eta_lambda = np.logspace(-5, 5, 1000)
    eta_omega = np.logspace(-5, 5, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1 * np.ones_like(eta_lambda_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m, eta_omega_grid, eta_lambda_grid
    )
    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, 2 * np.sqrt(uncertainty) - 1)
    mesh.set_clim(0, None)

    fig.colorbar(mesh)
    ax.set_title(
        r"Plot of quantum uncertainty against $\eta_\lambda$ and $\eta_\omega$"
    )
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.show()


def plot_uncertainty_against_lambda_w_lower() -> None:
    eta_lambda = np.linspace(0, 1, 1000)
    eta_omega = np.linspace(0.001, 1, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1 * np.ones_like(eta_lambda_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m, eta_omega_grid, eta_lambda_grid
    )

    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, 2 * np.sqrt(uncertainty) - 1)
    mesh.set_clim(0, None)

    # Plot the classical threshold
    def get_threshold_from_omega(
        eta_omega: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return (1 - 4 * eta_omega) ** 2 / 2

    def get_threshold_from_omega_lower(
        eta_omega: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return (2 - np.sqrt(2)) / np.abs(
            (8 - 6 * np.sqrt(2)) + ((2 * np.sqrt(2)) / (4 * eta_omega) ** 2)
        )

    old_lim = ax.get_xlim()
    (line,) = ax.plot(get_threshold_from_omega(eta_omega), eta_omega)
    line.set_label("Classical threshold")
    line.set_color("C5")
    line.set_linestyle("--")
    (line,) = ax.plot(get_threshold_from_omega_lower(eta_omega), eta_omega)
    line.set_color("C6")
    line.set_linestyle("--")
    ax.set_xlim(0, old_lim[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    fig.colorbar(mesh)
    ax.set_title(
        r"Plot of quantum uncertainty against $\eta_\lambda$ and $\eta_\omega$"
    )
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()


def plot_uncertainty_x_against_m_w() -> None:
    eta_m = np.linspace(1e20, 1e22, 1000)
    eta_omega = np.linspace(0.1, 4, 1000)
    eta_m_grid, eta_omega_grid = np.meshgrid(eta_m, eta_omega)
    eta_lambda = np.ones_like(eta_m_grid)
    uncertainty = (
        evaluate_equilibrium_expect_x_squared(eta_m_grid, eta_omega_grid, eta_lambda)
        * eta_m_grid
    )

    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_m_grid, eta_omega_grid, np.sqrt(uncertainty))
    mesh.set_norm(LogNorm())

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty x against $\eta_m$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\omega$")
    fig.show()


def plot_uncertainty_x_against_lambda_m() -> None:
    eta_m = np.linspace(1e20, 1e22, 1000)
    eta_lambda = np.linspace(1, 150, 1000)
    eta_m_grid, eta_lambda_grid = np.meshgrid(eta_m, eta_lambda)
    eta_omega = np.ones_like(eta_m_grid)
    uncertainty = (
        evaluate_equilibrium_expect_x_squared(eta_m_grid, eta_omega, eta_lambda_grid)
        * eta_m_grid
    )

    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_m_grid, eta_lambda_grid, np.sqrt(uncertainty))
    mesh.set_norm(LogNorm())

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty x against $\eta_m$ and $\eta_\lambda$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\lambda$")
    fig.show()


def plot_uncertainty_x_against_lambda_w() -> None:
    eta_lambda = np.linspace(0.001, 150, 1000)
    eta_omega = np.linspace(0.001, 20, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1 * np.ones_like(eta_lambda_grid)
    uncertainty = (
        evaluate_equilibrium_expect_x_squared(eta_m, eta_omega_grid, eta_lambda_grid)
        * eta_m
        / eta_omega_grid
    )

    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, uncertainty)
    mesh.set_clim(0, 1)

    for name, parameters, c in PHYSICAL_PARAMS:
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)  # type: ignore unknown
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty x against $\eta_\lambda$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()


def get_equilibrium_squeeze_ratio_free() -> sp.Expr:
    equilibrium = get_equilibrium_squeeze_ratio()
    return sp.limit(equilibrium, eta_omega, sp.oo)


def evaluate_equilibrium_uncertainty_free(
    eta_m: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    eta_lambda: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    expression_r = get_equilibrium_squeeze_ratio_free()
    lambda_r = sp.lambdify((_eta_m_symbol, _eta_lambda_symbol), expression_r)
    equilibrium_r = lambda_r(eta_m, eta_lambda)  # type: ignore unknown

    ratio_formula = sp.factor(squeeze_ratio_from_zeta_expr(uncertainty_squared))
    ratio_formula = ratio_formula.subs(
        {
            squeeze_ratio + sp.conjugate(squeeze_ratio): 2 * sp.re(squeeze_ratio),
            squeeze_ratio * sp.conjugate(squeeze_ratio): sp.Abs(squeeze_ratio) ** 2,
            hbar: 1,
        }
    )
    uncertainty_from_r = sp.lambdify((squeeze_ratio), ratio_formula)
    return np.real_if_close(uncertainty_from_r(equilibrium_r))  # type: ignore unknown


def plot_uncertainty_against_lambda_free_particle() -> None:
    eta_lambda = np.linspace(0.000001, 5, 10000)
    eta_m = 1 * np.ones_like(eta_lambda)
    uncertainty = evaluate_equilibrium_uncertainty_free(eta_m, eta_lambda)

    fig, ax = plot.get_figure()

    ax.plot(eta_lambda, 2 * np.sqrt(uncertainty) - 1)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 5)

    ax.set_title(r"Plot of uncertainty against $\eta_\lambda$ for a free particle")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"${2 \Delta - 1}$")
    fig.show()


def evaluate_equilibrium_uncertainty_x_free(
    eta_m: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    eta_lambda: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    expression_r = get_equilibrium_squeeze_ratio_free()
    lambda_r = sp.lambdify((_eta_m_symbol, _eta_lambda_symbol), expression_r)
    equilibrium_r = lambda_r(eta_m, eta_lambda)  # type: ignore unknown

    ratio_formula = sp.factor(squeeze_ratio_from_zeta_expr(uncertainty_squared))
    ratio_formula = ratio_formula.subs(
        {
            squeeze_ratio + sp.conjugate(squeeze_ratio): 2 * sp.re(squeeze_ratio),
            squeeze_ratio * sp.conjugate(squeeze_ratio): sp.Abs(squeeze_ratio) ** 2,
            hbar: 1,
        }
    )
    uncertainty_from_r = sp.lambdify((squeeze_ratio), ratio_formula)
    return np.real_if_close(uncertainty_from_r(equilibrium_r))  # type: ignore unknown


def plot_uncertainty_x_against_lambda_free_particle() -> None:
    eta_lambda = np.linspace(0.000001, 20, 10000)
    eta_m = 1 * np.ones_like(eta_lambda)
    uncertainty = evaluate_equilibrium_uncertainty_x_free(eta_m, eta_lambda)

    fig, ax = plot.get_figure()

    def low_friction(
        eta_lambda: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        return (np.sqrt(2) * np.sqrt(eta_lambda)) / 2

    (line,) = ax.plot(eta_lambda, uncertainty)
    line.set_label("Free particle")
    (line,) = ax.plot(eta_lambda, low_friction(eta_lambda))
    line.set_label("Low friction limit")
    line.set_linestyle("--")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")

    ax.set_title(r"Plot of uncertainty in x against $\eta_\lambda$ with $\eta_m$ = 1")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"Uncertainty in x")
    fig.show()


if __name__ == "__main__":
    plot_uncertainty_against_lambda_w()
    plot_uncertainty_against_lambda_w_logspace()
    plot_uncertainty_against_lambda_w_lower()
    plot_uncertainty_against_lambda_free_particle()
    plot_uncertainty_x_against_lambda_free_particle()
    plot_uncertainty_x_against_lambda_w()

    plot.wait_for_close()
