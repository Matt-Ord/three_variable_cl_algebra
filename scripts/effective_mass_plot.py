from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from slate_core import plot
from sympy.physics.units import hbar

from three_variable.equilibrium_squeeze import get_equilibrium_zeta
from three_variable.projected_sse import get_classical_deterministic_derivative
from three_variable.symbols import KBT, eta_lambda, eta_m, eta_omega, noise, p, x, zeta

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_lambda_omega_formula(
    formula: sp.Expr, *, measure: plot.Measure = "abs"
) -> tuple[Figure, Axes]:
    eta_omega_value = np.logspace(-8, 8, 500)
    eta_lambda_value = np.logspace(-8, 8, 500)

    zeta_equilibrium = get_equilibrium_zeta().subs({eta_m: 1})
    zeta_fn = sp.lambdify(
        (eta_lambda, eta_omega),
        zeta_equilibrium,
        modules="numpy",
    )
    formula_fn = sp.lambdify(
        (eta_lambda, eta_omega, zeta),
        formula,
        modules="numpy",
    )

    l_v, o_v = np.meshgrid(eta_lambda_value, eta_omega_value, indexing="xy")

    fig, ax = plot.get_figure()
    mesh = ax.pcolormesh(
        eta_lambda_value,
        eta_omega_value,
        plot.get_measured_data(formula_fn(l_v, o_v, zeta_fn(l_v, o_v)), measure),  # type: ignore args
        cmap="viridis",
    )
    fig.colorbar(mesh)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    return fig, ax


def plot_effective_frequency() -> None:
    eta_m_val = 1
    p_derivative_x = get_classical_deterministic_derivative("p").subs(
        {x: 1, p: 0, noise: 0, KBT: 1, hbar: 1, eta_m: eta_m_val, sp.Symbol("V_1"): 0}
    )
    mass_fraction = sp.simplify(p_derivative_x * eta_omega**2 / eta_m_val)

    fig, ax = plot_lambda_omega_formula(mass_fraction, measure="abs")
    ax.set_title("Effective Frequency (Absolute Value)")
    fig.show()


def plot_zeta() -> None:
    fig, ax = plot_lambda_omega_formula(zeta)
    ax.set_title("Zeta")
    fig.show()

    fig, ax = plot_lambda_omega_formula(zeta, measure="real")
    ax.set_title("Zeta (Real Part)")
    fig.show()

    fig, ax = plot_lambda_omega_formula(zeta, measure="imag")
    ax.set_title("Zeta (Imaginary Part)")
    fig.show()


def plot_effective_mass() -> None:
    x_derivative_p = get_classical_deterministic_derivative("x").subs(
        {x: 0, p: 1, noise: 0, KBT: 1, hbar: 1, eta_m: 1, sp.Symbol("V_1"): 0}
    )
    mass_fraction = sp.simplify(2 / x_derivative_p)

    fig, ax = plot_lambda_omega_formula(mass_fraction, measure="abs")
    ax.set_title("Effective Mass (Absolute Value)")
    fig.show()


def plot_effective_friction() -> None:
    p_derivative_p = get_classical_deterministic_derivative("p").subs(
        {x: 0, p: 1, noise: 0, KBT: 1, hbar: 1, eta_m: 1, sp.Symbol("V_1"): 0}
    )
    lambda_fraction = sp.simplify(p_derivative_p * eta_lambda)

    fig, ax = plot_lambda_omega_formula(lambda_fraction, measure="abs")
    ax.set_title("Effective Friction (Absolute Value)")
    fig.show()


def plot_x_force() -> None:
    kb_t_val = 1
    hbar_val = 1
    x_derivative_x = get_classical_deterministic_derivative("x").subs(
        {
            x: 1,
            p: 0,
            noise: 0,
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: 1,
            sp.Symbol("V_1"): 0,
        }
    )
    lambda_fraction = sp.simplify(x_derivative_x * hbar_val / kb_t_val)

    fig, ax = plot_lambda_omega_formula(lambda_fraction, measure="abs")
    ax.set_title("X Force (Absolute Value)")
    fig.show()


if __name__ == "__main__":
    plot_effective_friction()
    plot_zeta()
    plot_effective_frequency()
    plot_effective_mass()
    plot_x_force()

    plot.wait_for_close()
