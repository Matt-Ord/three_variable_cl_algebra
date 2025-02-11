from __future__ import annotations

import numpy as np
import sympy as sp

from three_variable import (
    util,
)
from three_variable.equilibrium_squeeze import (
    R,
    evaluate_equilibrium_beta,
    evaluate_equilibrium_R,
    get_equilibrium_squeeze_R,
    get_uncertainty_R,
    get_uncertainty_x_R,
)
from three_variable.equilibrium_squeeze import eta_lambda as _eta_lambda_symbol
from three_variable.equilibrium_squeeze import eta_m as _eta_m_symbol
from three_variable.physical_systems import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU
from three_variable.symbols import eta_omega

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]


def plot_beta_against_m_w() -> None:
    # Plot of beta against eta_m and eta_omega with low friction
    eta_m = np.linspace(0.0001, 2, 100)
    eta_omega = np.linspace(0.0001, 2, 1000)
    eta_m_grid, eta_omega_grid = np.meshgrid(eta_m, eta_omega)
    eta_lambda = 10 * np.ones_like(eta_m_grid)
    beta = evaluate_equilibrium_beta(eta_m_grid, eta_omega_grid, eta_lambda)

    fig, ax = util.get_figure()
    mesh = ax.pcolormesh(eta_m_grid, eta_omega_grid, np.abs(beta))

    fig.colorbar(mesh)
    ax.set_title(r"Plot of $|{\beta}|$ against $\eta_m$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\omega$")
    fig.show()


def plot_beta_against_lambda_w() -> None:
    # Plot of beta against eta_m and eta_omega with high mass
    eta_lambda = np.linspace(0.01, 150, 1000)
    eta_omega = np.linspace(0.01, 10, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 10 * np.ones_like(eta_lambda_grid)
    beta = evaluate_equilibrium_beta(eta_m, eta_omega_grid, eta_lambda_grid)

    fig, ax = util.get_figure()
    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, np.abs(beta))

    for name, parameters, c in PHYSICAL_PARAMS:
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of $|{\beta}|$ against $\eta_\lambda$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()

    fig, ax = util.get_figure()
    theta = np.angle(beta)
    # theta = np.mod(theta + 2 * np.pi, 2 * np.pi)
    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, theta)

    for name, parameters, c in PHYSICAL_PARAMS:
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of $arg{\beta}$ against $\eta_\lambda$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()


def evaluate_equilibrium_uncertainty(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    equilibrium_R = evaluate_equilibrium_R(  # noqa: N806
        eta_m, eta_omega, eta_lambda, positive=positive
    )

    uncertainty_from_R = sp.lambdify((R), get_uncertainty_R())
    return np.real_if_close(uncertainty_from_R(equilibrium_R))


def plot_uncertainty_against_m_w() -> None:
    eta_m = np.linspace(0.0001, 2, 1000)
    eta_omega = np.linspace(0.0001, 10, 1000)
    eta_m_grid, eta_omega_grid = np.meshgrid(eta_m, eta_omega)
    eta_lambda = np.ones_like(eta_m_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m_grid, eta_omega_grid, eta_lambda
    )

    fig, ax = util.get_figure()

    mesh = ax.pcolormesh(eta_m_grid, eta_omega_grid, uncertainty)
    mesh.set_norm(util.LogNorm())
    mesh.set_clim(1 / 16, None)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty against $\eta_m$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\omega$")
    fig.show()


def plot_uncertainty_against_lambda_w() -> None:
    eta_lambda = np.linspace(0.001, 150, 1000)
    eta_omega = np.linspace(0.001, 20, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1 * np.ones_like(eta_lambda_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m, eta_omega_grid, eta_lambda_grid
    )

    fig, ax = util.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, 4 * np.sqrt(uncertainty) - 1)
    mesh.set_clim(0, None)

    for name, parameters, c in PHYSICAL_PARAMS:
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty against $\eta_\lambda$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()


def evaluate_equilibrium_uncertainty_x(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    equilibrium_R = evaluate_equilibrium_R(  # noqa: N806
        eta_m, eta_omega, eta_lambda, positive=positive
    )

    uncertainty_from_R = sp.lambdify((R), get_uncertainty_x_R())
    return np.real_if_close(uncertainty_from_R(equilibrium_R))


def plot_uncertainty_x_against_m_w() -> None:
    eta_m = np.linspace(1e20, 1e22, 1000)
    eta_omega = np.linspace(0.1, 4, 1000)
    eta_m_grid, eta_omega_grid = np.meshgrid(eta_m, eta_omega)
    eta_lambda = np.ones_like(eta_m_grid)
    uncertainty = evaluate_equilibrium_uncertainty_x(
        eta_m_grid, eta_omega_grid, eta_lambda
    )

    fig, ax = util.get_figure()

    mesh = ax.pcolormesh(eta_m_grid, eta_omega_grid, np.sqrt(uncertainty))
    mesh.set_norm(util.LogNorm())

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
    uncertainty = evaluate_equilibrium_uncertainty_x(
        eta_m_grid, eta_omega, eta_lambda_grid
    )

    fig, ax = util.get_figure()

    mesh = ax.pcolormesh(eta_m_grid, eta_lambda_grid, np.sqrt(uncertainty))
    mesh.set_norm(util.LogNorm())

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty x against $\eta_m$ and $\eta_\lambda$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\lambda$")
    fig.show()


def plot_uncertainty_x_against_lambda_w() -> None:
    eta_lambda = np.linspace(0.001, 150, 1000)
    eta_omega = np.linspace(0.001, 20, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1e21 * np.ones_like(eta_lambda_grid)
    uncertainty = evaluate_equilibrium_uncertainty_x(
        eta_m, eta_omega_grid, eta_lambda_grid
    )

    fig, ax = util.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, np.sqrt(uncertainty))
    # mesh.set_norm(util.LogNorm())

    for name, parameters, c in PHYSICAL_PARAMS:
        # print(parameters.eta_lambda, parameters.eta_omega, parameters.eta_m)
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty x against $\eta_\lambda$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    ax.legend(loc="upper right")
    fig.show()


def get_equilibrium_squeeze_R_free() -> sp.Expr:
    equilibrium = get_equilibrium_squeeze_R()
    return sp.limit(equilibrium, eta_omega, np.inf)


def evaluate_equilibrium_uncertainty_free(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    expression_R = get_equilibrium_squeeze_R_free()
    lambda_R = sp.lambdify((_eta_m_symbol, _eta_lambda_symbol), expression_R)
    equilibrium_R = lambda_R(eta_m, eta_lambda)

    uncertainty_from_R = sp.lambdify((R), get_uncertainty_R())
    return np.real_if_close(uncertainty_from_R(equilibrium_R))


def plot_uncertainty_against_lambda_free_particle() -> None:
    eta_lambda = np.linspace(0.001, 150, 1000)
    eta_m = 1 * np.ones_like(eta_lambda)
    uncertainty = evaluate_equilibrium_uncertainty_free(eta_m, eta_lambda)

    fig, ax = util.get_figure()

    ax.plot(eta_lambda, 4 * np.sqrt(uncertainty) - 1)
    # mesh.set_clim(0, None)

    # for name, parameters, c in PHYSICAL_PARAMS:
    #     scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)
    #     scatter.set_label(name)
    #     scatter.set_color(c)

    # fig.colorbar(mesh)
    ax.set_title(r"Plot of uncertainty against $\eta_\lambda$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"${4 \Delta^2 - 1}$")
    fig.show()


if __name__ == "__main__":
    plot_beta_against_m_w()
    plot_beta_against_lambda_w()
    plot_uncertainty_against_lambda_w()
    plot_uncertainty_against_m_w()
    plot_uncertainty_against_lambda_free_particle()
    plot_uncertainty_x_against_m_w()
    plot_uncertainty_x_against_lambda_w()
    plot_uncertainty_x_against_lambda_m()

    util.wait_for_close()
