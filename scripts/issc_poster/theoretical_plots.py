from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from slate import plot

from three_variable.equilibrium_squeeze import (
    evaluate_equilibrium_expect_x_squared,
    evaluate_equilibrium_uncertainty,
)
from three_variable.physical_systems import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU

mpl.rcParams["axes.labelsize"] = 14

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]


print("Omega      | m          | lambda        | uncertainty")
print(
    f"{TOWNSEND_H_RU.omega:0.3e}",
    TOWNSEND_H_RU.m,
    f"{TOWNSEND_H_RU.lambda_:0.3e}",
    evaluate_equilibrium_uncertainty(
        TOWNSEND_H_RU.eta_parameters.eta_m,  # type: ignore sp
        TOWNSEND_H_RU.eta_parameters.eta_omega,  # type: ignore sp
        TOWNSEND_H_RU.eta_parameters.eta_lambda,  # type: ignore sp
    ),
)
print(
    f"{ELENA_LI_CU.omega:0.3e}",
    ELENA_LI_CU.m,
    f"{ELENA_LI_CU.lambda_:0.3e}",
    evaluate_equilibrium_uncertainty(
        ELENA_LI_CU.eta_parameters.eta_m,  # type: ignore sp
        ELENA_LI_CU.eta_parameters.eta_omega,  # type: ignore sp
        ELENA_LI_CU.eta_parameters.eta_lambda,  # type: ignore sp
    ),
)
print(
    f"{ELENA_NA_CU.omega:0.3e}",
    ELENA_NA_CU.m,
    f"{ELENA_NA_CU.lambda_:0.3e}",
    evaluate_equilibrium_uncertainty(
        ELENA_NA_CU.eta_parameters.eta_m,  # type: ignore sp
        ELENA_NA_CU.eta_parameters.eta_omega,  # type: ignore sp
        ELENA_NA_CU.eta_parameters.eta_lambda,  # type: ignore sp
    ),
)


def plot_uncertainty_against_lambda_w_lower() -> None:
    eta_lambda = np.linspace(0, 1, 2000)
    eta_omega = np.linspace(0.0001, 1, 2000)
    eta_lambda_grid, eta_omega_grid = np.meshgrid(eta_lambda, eta_omega)
    eta_m = 1 * np.ones_like(eta_lambda_grid)
    uncertainty = evaluate_equilibrium_uncertainty(
        eta_m, eta_omega_grid, eta_lambda_grid
    )

    fig, ax = plot.get_figure()

    mesh = ax.pcolormesh(eta_lambda_grid, eta_omega_grid, 2 * np.sqrt(uncertainty) - 1)
    mesh.set_clim(0, None)

    # Plot the classical threshold
    def get_threshold_from_omega_upper(
        eta_omega: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return np.abs(16 * eta_omega**2 + 8 * eta_omega - 1) / 4

    def get_threshold_from_omega_lower(
        eta_omega: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return (2 - np.sqrt(2)) / np.abs(
            4 + ((2 * np.sqrt(2) - 3) / (2 * eta_omega) ** 2)
        )

    upper = get_threshold_from_omega_upper(eta_omega)
    lower = get_threshold_from_omega_lower(eta_omega)

    ignore = upper < lower
    upper[ignore] = np.nan
    lower[ignore] = np.nan
    old_lim = ax.get_xlim()
    (line,) = ax.plot(upper, eta_omega)
    line.set_label("Transition")
    line.set_color("C7")
    line.set_linestyle("--")
    line.set_linewidth(2)
    (line,) = ax.plot(lower, eta_omega)
    line.set_color("C7")
    line.set_linestyle("--")
    line.set_linewidth(2)
    ax.set_xlim(0, old_lim[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    fig.colorbar(mesh)
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$\Omega$")
    ax.legend(loc="upper right")
    fig.set_size_inches(8, 5.5)
    fig.tight_layout()
    fig.savefig(
        f"{Path(__file__).parent}/out/theoretical_uncertainty.png",
        dpi=600,  # type: ignore sp
        facecolor="none",  # type: ignore sp
        transparent=True,
    )

    fig.show()


def plot_uncertainty_x_against_lambda_w() -> None:
    eta_lambda = np.linspace(0.001, 150, 2000)
    eta_omega = np.linspace(0.001, 20, 2000)
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
        scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)  # type: ignore sp
        scatter.set_label(name)
        scatter.set_color(c)

    fig.colorbar(mesh)
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$\Omega$")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, ax.get_ylim()[1])
    fig.tight_layout()
    fig.set_size_inches(8, 5.5)
    fig.tight_layout()
    fig.savefig(
        f"{Path(__file__).parent}/out/theoretical_width.png",
        dpi=600,  # type: ignore sp
        facecolor="none",  # type: ignore sp
        transparent=True,
    )
    fig.show()


if __name__ == "__main__":
    plot_uncertainty_against_lambda_w_lower()
    plot_uncertainty_x_against_lambda_w()

    plot.wait_for_close()
