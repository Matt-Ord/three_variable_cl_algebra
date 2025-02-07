from __future__ import annotations

import numpy as np
import sympy as sp

from three_variable import (
    util,
)
from three_variable.equilibrium_squeeze import (
    eta_lambda,
    eta_m,
    eta_omega,
    get_equilibrium_squeeze_derivative_gradient,
)

_eta_m_symbol = eta_m
_eta_omega_symbol = eta_omega
_eta_lambda_symbol = eta_lambda


def evaluate_equilibrium_squeeze_derivative_gradient(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    expression = get_equilibrium_squeeze_derivative_gradient(positive=positive)
    expression_lambda = sp.lambdify(
        (_eta_m_symbol, _eta_omega_symbol, _eta_lambda_symbol), expression
    )
    return expression_lambda(eta_m, eta_omega, eta_lambda)


def plot_beta_against_m_w() -> None:
    # Plot of beta against eta_m and eta_omega with low friction
    eta_m = np.linspace(0.01, 2, 100)
    eta_omega = np.linspace(0.01, 2, 1000)
    eta_m_grid, eta_omega_grid = np.meshgrid(eta_m, eta_omega)
    eta_lambda = 10 * np.ones_like(eta_m_grid)
    gradient = evaluate_equilibrium_squeeze_derivative_gradient(
        eta_m_grid, eta_omega_grid, eta_lambda
    )

    fig, ax = util.get_figure()
    mesh = ax.pcolormesh(eta_m_grid, eta_omega_grid, np.real(gradient))
    mesh.set_clim(-1, -0)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of real gradient against $\eta_m$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_m$")
    ax.set_ylabel(r"$\eta_\omega$")
    fig.show()


def plot_beta_against_lambda_w() -> None:
    # Plot of beta against eta_m and eta_omega with high mass
    eta_lambda = np.linspace(0.01, 150, 1000)
    eta_omega = np.linspace(0.01, 10, 1000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 10 * np.ones_like(eta_lambda_grid)
    gradient = evaluate_equilibrium_squeeze_derivative_gradient(
        eta_m, eta_omega_grid, eta_lambda_grid
    )

    fig, ax = util.get_figure()
    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, np.real(gradient))
    mesh.set_clim(-1, -0)

    fig.colorbar(mesh)
    ax.set_title(r"Plot of real gradient against $\eta_\lambda$ and $\eta_\omega$")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    fig.show()


if __name__ == "__main__":
    plot_beta_against_m_w()
    plot_beta_against_lambda_w()

    util.wait_for_close()
