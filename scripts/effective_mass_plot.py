from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from classical_equation_of_motion import (
    get_classical_equilibrium_derivative_ratio,
    get_high_mass_equilibrium_value,
    substitute_back_r0,
)
from matplotlib.colors import SymLogNorm
from scipy.constants import Boltzmann  # type: ignore scipy
from slate_core import plot
from sympy.physics.units import hbar

from three_variable.equilibrium_squeeze import get_equilibrium_zeta
from three_variable.simulation import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU
from three_variable.symbols import KBT, eta_lambda, eta_m, eta_omega, noise, p, x

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure


PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]


def plot_lambda_omega_formula_high_mass(
    formula: sp.Expr, *, measure: plot.Measure = "real"
) -> tuple[Figure, Axes, QuadMesh]:
    eta_omega_value = np.logspace(-10, 10, 500)
    eta_lambda_value = np.logspace(-10, 10, 500)
    formula_fn = sp.lambdify(
        (eta_lambda, eta_omega),
        formula,
        modules="numpy",
    )

    l_v, o_v = np.meshgrid(eta_lambda_value, eta_omega_value, indexing="xy")

    fig, ax = plot.get_figure()
    mesh = ax.pcolormesh(
        eta_lambda_value,
        eta_omega_value,
        plot.get_measured_data(formula_fn(l_v, o_v), measure),  # type: ignore args
        cmap="viridis",
    )
    # for name, parameters, c in PHYSICAL_PARAMS:
    #     scatter = ax.scatter(parameters.eta_lambda, parameters.eta_omega)  # type: ignore unknown
    #     scatter.set_label(name)
    #     scatter.set_color(c)
    fig.colorbar(mesh, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta_\lambda$")
    ax.set_ylabel(r"$\eta_\omega$")
    # ax.legend(loc="upper right")
    return fig, ax, mesh


def plot_effective_frequency_high_mass() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    p_derivative_x = get_classical_equilibrium_derivative_ratio(
        "p", part="deterministic"
    ).subs(
        {
            x: 1,
            p: 0,
            noise: 0,
            sp.Symbol("V_1"): 0,
        }
    )
    p_derivative_x_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(p_derivative_x)
    )
    p_derivative_x_high_mass = p_derivative_x_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    x_derivative_p = get_classical_equilibrium_derivative_ratio(
        "x", part="deterministic"
    ).subs(
        {
            x: 0,
            p: 1,
            noise: 0,
            sp.Symbol("V_1"): 0,
        }
    )
    x_derivative_p_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(x_derivative_p)
    )
    x_derivative_p_high_mass = x_derivative_p_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    omega_fraction = (
        sp.sqrt(-p_derivative_x_high_mass * x_derivative_p_high_mass)
        * hbar_val
        * eta_omega
        / kb_t_val
    )

    fig, ax, mesh = plot_lambda_omega_formula_high_mass(omega_fraction, measure="real")
    # mesh.set_norm(SymLogNorm(linthresh=1, linscale=5))
    mesh.set_clim(0, None)
    ax.set_title("Effective Frequency (Real Value)")
    fig.savefig("effective_frequency.png", dpi=300)
    fig.show()


def plot_r0() -> None:
    r0 = get_equilibrium_zeta().subs({eta_m: 1})
    fig, ax, _mesh = plot_lambda_omega_formula_high_mass(r0, measure="abs")
    ax.set_title("Zeta")
    fig.savefig("r0_abs.png", dpi=300)
    fig.show()

    fig, ax, _mesh = plot_lambda_omega_formula_high_mass(r0, measure="real")
    ax.set_title("Zeta (Real Part)")
    fig.savefig("r0_real.png", dpi=300)
    fig.show()

    fig, ax, _mesh = plot_lambda_omega_formula_high_mass(r0, measure="imag")
    ax.set_title("Zeta (Imaginary Part)")
    fig.savefig("r0_imag.png", dpi=300)
    fig.show()


def plot_effective_mass_high_mass() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    x_derivative_p = get_classical_equilibrium_derivative_ratio(
        "x", part="deterministic"
    ).subs(
        {
            x: 0,
            p: 1,
            noise: 0,
            sp.Symbol("V_1"): 0,
        }
    )
    x_derivative_p_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(x_derivative_p)
    )
    x_derivative_p_high_mass = x_derivative_p_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    mass_fraction = 2 * kb_t_val / (hbar_val**2 * eta_m_val * x_derivative_p_high_mass)

    fig, ax, _mesh = plot_lambda_omega_formula_high_mass(mass_fraction, measure="real")
    # mesh.set_norm(SymLogNorm(linthresh=10, linscale=5))
    # mesh.set_clim(0, None)
    ax.set_title("Effective Mass (Real Value)")
    fig.savefig("effective_mass.png", dpi=300)
    fig.show()


def plot_effective_friction_high_mass() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    p_derivative_p = get_classical_equilibrium_derivative_ratio(
        "p", part="deterministic"
    ).subs(
        {
            x: 0,
            p: 1,
            noise: 0,
            sp.Symbol("V_1"): 0,
        }
    )
    p_derivative_p_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(p_derivative_p)
    )
    p_derivative_p_high_mass = p_derivative_p_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    lambda_fraction = -p_derivative_p_high_mass * eta_lambda * hbar_val / kb_t_val / 2

    fig, ax, _mesh = plot_lambda_omega_formula_high_mass(
        lambda_fraction, measure="real"
    )
    # mesh.set_norm(SymLogNorm(linthresh=10, linscale=5))
    # mesh.set_clim(0, None)
    ax.set_title("Effective Friction (Real Value)")
    fig.savefig("effective_friction.png", dpi=300)
    fig.show()


def plot_x_force_high_mass() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    x_derivative_x = get_classical_equilibrium_derivative_ratio(
        "x", part="deterministic"
    ).subs(
        {
            x: 1,
            p: 0,
            noise: 0,
            sp.Symbol("V_1"): 0,
        }
    )
    x_derivative_x_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(x_derivative_x)
    )
    x_derivative_x_high_mass = x_derivative_x_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    # lambda_fraction = sp.simplify(x_derivative_x_high_mass)
    lambda_fraction = x_derivative_x_high_mass
    fig, ax, mesh = plot_lambda_omega_formula_high_mass(lambda_fraction, measure="real")
    mesh.set_norm(SymLogNorm(linthresh=10, linscale=5))
    # mesh.set_clim(0, None)
    ax.set_title("X Force (Real Value)")
    fig.savefig("x_force.png", dpi=300)
    fig.show()


def plot_x_stochastic() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    print("getting x stochastic force")
    x_stochastic = get_classical_equilibrium_derivative_ratio(
        "x", part="stochastic"
    ).subs(
        {
            x: 0,
            p: 0,
            noise: 1j,
            sp.Symbol("V_1"): 0,
        }
    )
    sp.print_latex(x_stochastic)
    print("expanding and substituting back r0")
    x_stochastic_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(x_stochastic, expanding=False)
    )
    x_stochastic_high_mass = x_stochastic_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    print("lamdifying")
    # lambda_fraction = sp.simplify(x_derivative_x_high_mass)
    lambda_fraction = x_stochastic_high_mass
    fig, ax, mesh = plot_lambda_omega_formula_high_mass(lambda_fraction, measure="abs")
    mesh.set_norm(SymLogNorm(linthresh=10, linscale=5))
    # mesh.set_clim(0, None)
    ax.set_title("x stochastic force (Abs Value)")
    fig.savefig("x_stochastic.png", dpi=300)
    fig.show()


def plot_p_stochastic() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    p_stochastic = get_classical_equilibrium_derivative_ratio(
        "p", part="stochastic"
    ).subs(
        {
            x: 0,
            p: 0,
            noise: 1j,
            sp.Symbol("V_1"): 0,
        }
    )
    p_stochastic_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(p_stochastic, expanding=False)
    )
    p_stochastic_high_mass = p_stochastic_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    # lambda_fraction = sp.simplify(x_derivative_x_high_mass)
    lambda_fraction = p_stochastic_high_mass
    fig, ax, mesh = plot_lambda_omega_formula_high_mass(lambda_fraction, measure="abs")
    mesh.set_norm(SymLogNorm(linthresh=10, linscale=5))
    # mesh.set_clim(0, None)
    ax.set_title("p stochastic force (Abs Value)")
    fig.savefig("p_stochastic.png", dpi=300)
    fig.show()


def plot_p_fluctuation() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    p_stochastic = get_classical_equilibrium_derivative_ratio(
        "p", part="stochastic"
    ).subs(
        {
            x: 0,
            p: 0,
            noise: (1 + 1j) / sp.sqrt(2),
            sp.Symbol("V_1"): 0,
        }
    )
    p_stochastic_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(p_stochastic, expanding=False)
    )
    p_stochastic_high_mass = p_stochastic_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    # lambda_fraction = sp.simplify(x_derivative_x_high_mass)
    # p_rms = sp.sqrt(eta_m_val * hbar_val**2 / 2)
    lambda_fraction = (
        p_stochastic_high_mass
        * sp.conjugate(p_stochastic_high_mass)
        * eta_lambda
        / (2 * eta_m_val * hbar_val * kb_t_val)
    )
    fig, ax, mesh = plot_lambda_omega_formula_high_mass(lambda_fraction, measure="abs")
    mesh.set_norm(SymLogNorm(linthresh=0.01, linscale=1))
    # mesh.set_clim(0.01, 1)
    ax.set_title("p mean squared fluctuation")
    fig.savefig("p_fluctuation.png", dpi=300)
    fig.show()


def plot_x_fluctuation() -> None:
    kb_t_val = Boltzmann * 300
    hbar_val = 1.0545718e-34
    eta_m_val = ELENA_NA_CU.eta_parameters.eta_m
    x_stochastic = get_classical_equilibrium_derivative_ratio(
        "x", part="stochastic"
    ).subs(
        {
            x: 0,
            p: 0,
            noise: (1 + 1j) / sp.sqrt(2),
            sp.Symbol("V_1"): 0,
        }
    )
    x_stochastic_high_mass = substitute_back_r0(
        get_high_mass_equilibrium_value(x_stochastic, expanding=False)
    )
    x_stochastic_high_mass = x_stochastic_high_mass.subs(
        {
            KBT: kb_t_val,
            hbar: hbar_val,
            eta_m: eta_m_val,
        }
    )
    # lambda_fraction = sp.simplify(x_derivative_x_high_mass)
    # x_rms = sp.sqrt(2 * eta_omega**2 / eta_m_val)
    p_over_m = sp.sqrt(2 / eta_m_val) * (kb_t_val / hbar_val)
    lambda_fraction = x_stochastic_high_mass
    fig, ax, mesh = plot_lambda_omega_formula_high_mass(
        lambda_fraction / p_over_m, measure="abs"
    )
    mesh.set_norm(SymLogNorm(linthresh=1e-15, linscale=1))
    # mesh.set_clim(1e-9, None)
    ax.set_title("x noise / p/m)")
    fig.savefig("x_fluctuation.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    plot_r0()
    plot_x_stochastic()
    plot_p_stochastic()
    plot_x_fluctuation()
    plot_p_fluctuation()
    plot_effective_friction_high_mass()
    plot_effective_frequency_high_mass()
    plot_effective_mass_high_mass()
    plot_x_force_high_mass()

    plot.wait_for_close()
