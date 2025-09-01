from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import sympy as sp
from scipy.constants import (  # type: ignore scipy
    Boltzmann,
)
from slate_core import plot
from slate_core.util import timed
from sympy.physics.units import hbar

from three_variable.equilibrium_squeeze import (
    get_equilibrium_squeeze_ratio,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.fokker_planck import (
    B,
    X_force,
    b11,
    b12,
    b21,
    b22,
    derive_fokker_planck,
    drift,
    friction,
    gaussian_ansatz,
    inertia,
    p,
    s11,
    s12,
    s22,
    stiffness,
    x,
)
from three_variable.projected_sse import (
    get_deterministic_derivative,
    get_stochastic_derivative,
)
from three_variable.simulation import (
    ELENA_NA_CU,
    EtaParameters,
)
from three_variable.symbols import KBT, alpha, eta_lambda, eta_m, eta_omega, noise
from three_variable.util import file_cached

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure

from three_variable.coherent_states import (
    expectation_from_formula,
)
from three_variable.equilibrium_squeeze import (
    evaluate_equilibrium_squeeze_ratio,
)
from three_variable.projected_sse import get_harmonic_term, get_kinetic_term
from three_variable.simulation import (
    hbar_value,
)
from three_variable.symbols import (
    dimensionless_from_full,
    formula_from_expr,
)

KBT_value = Boltzmann * ELENA_NA_CU.temperature * 2
# note hbar=KBT when calculating inertia, stiffness, etc. hbar value in SI unit used in energy calculation.
eta_params = EtaParameters(
    eta_lambda=60,
    eta_omega=2,
    eta_m=1e22,
    kbt_div_hbar=KBT_value / hbar_value,
)

# alpha_prime
alpha_prime = sp.Symbol(r"\alpha'", complex=True)
alpha_prime_expr = (squeeze_ratio + 1 / eta_m) / (2 - squeeze_ratio) * alpha


def alpha_prime_from_alpha(expr: sp.Expr) -> sp.Expr:
    """Convert an expression in terms of alpha to an expression in terms of real and imaginary part of alpha_prime."""
    expr = sp.simplify(
        expr.subs(
            {alpha: alpha_prime * (2 - squeeze_ratio) / (squeeze_ratio + 1 / eta_m)}
        )
    )
    return sp.simplify(
        expr.subs(alpha_prime, sp.re(alpha_prime) + 1j * sp.im(alpha_prime))
    )


def _get_transformed_deterministic_derivative(
    ty: Literal["x_like", "p_like"],
) -> Path:
    return Path(f".cache/transformed_deterministic_derivative.{ty}")


@file_cached(_get_transformed_deterministic_derivative)
@timed
def get_transformed_deterministic_derivative(
    ty: Literal["x_like", "p_like"],
) -> sp.Expr:
    # Write deterministic derivative of alpha in terms of real and imaginary parts of alpha prime.
    # Then replace re(alpha_prime) and im(alpha_prime) with their derivatives.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative = alpha_prime_from_alpha(
        squeeze_ratio_from_zeta_expr(get_deterministic_derivative("alpha"))
    )
    alpha_prime_derivative = sp.simplify(
        alpha_prime_expr.subs(
            {
                alpha: derivative,
                sp.conjugate(alpha): sp.conjugate(derivative),
            }
        ),
        rational=True,
    )
    expect_var = (
        sp.re(alpha_prime_derivative)
        if ty == "x_like"
        else sp.im(alpha_prime_derivative)
    )
    return sp.simplify(
        expect_var,
        rational=True,
    )


def _get_transformed_stochastic_derivative(
    ty: Literal["x_like", "p_like"],
) -> Path:
    return Path(f".cache/transformed_stochastic_derivative.{ty}")


@file_cached(_get_transformed_stochastic_derivative)
@timed
def get_transformed_stochastic_derivative(
    ty: Literal["x_like", "p_like"],
) -> sp.Expr:
    # Write stochastic derivative of alpha in terms of real and imaginary parts of alpha prime.
    # Then replace re(alpha_prime) and im(alpha_prime) with their derivatives.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative = alpha_prime_from_alpha(
        squeeze_ratio_from_zeta_expr(get_stochastic_derivative("alpha"))
    )

    alpha_prime_derivative = sp.simplify(
        alpha_prime_expr.subs(
            {
                alpha: derivative,
                sp.conjugate(alpha): sp.conjugate(derivative),
            }
        ),
        rational=True,
    )
    expect_var = (
        sp.re(alpha_prime_derivative)
        if ty == "x_like"
        else sp.im(alpha_prime_derivative)
    )
    return sp.simplify(
        expect_var,
        rational=True,
    )


def get_squeeze_ratio_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    ratio_fn = sp.lambdify(
        (eta_lambda, eta_omega),
        get_equilibrium_squeeze_ratio(),
        modules="numpy",
    )
    return ratio_fn(eta_param.eta_lambda, eta_param.eta_omega)


def get_inertia_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    # dx_like / dt = inertia * p_like + X_force * x_like
    x_derivative_p_coeff = get_transformed_deterministic_derivative("x_like").subs(
        {alpha_prime: 1j, sp.Symbol("V_1"): 0, KBT: KBT_value, hbar: KBT_value}
    )  # set hbar = kbt does not change the probability distribution
    inertia_fn = sp.lambdify(
        (squeeze_ratio, eta_m, eta_lambda, eta_omega),
        x_derivative_p_coeff,
        modules="numpy",
    )
    return inertia_fn(
        get_squeeze_ratio_value(eta_param),
        eta_param.eta_m,
        eta_param.eta_lambda,
        eta_param.eta_omega,
    )


def get_stiffness_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    # d p_like / dt = -stiffness * x_like - friction * p_like
    p_derivative_x_coeff = get_transformed_deterministic_derivative("p_like").subs(
        {alpha_prime: 1, sp.Symbol("V_1"): 0, KBT: KBT_value, hbar: KBT_value}
    )
    stiffness_fn = sp.lambdify(
        (squeeze_ratio, eta_m, eta_lambda, eta_omega),
        -p_derivative_x_coeff,
        modules="numpy",
    )
    return stiffness_fn(
        get_squeeze_ratio_value(eta_param),
        eta_param.eta_m,
        eta_param.eta_lambda,
        eta_param.eta_omega,
    )


def get_x_force_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    # dx_like / dt = inertia * p_like + X_force * x_like
    x_force_expr = get_transformed_deterministic_derivative("x_like").subs(
        {alpha_prime: 1, sp.Symbol("V_1"): 0, KBT: KBT_value, hbar: KBT_value}
    )
    x_force_fn = sp.lambdify(
        (squeeze_ratio, eta_m, eta_lambda, eta_omega),
        x_force_expr,
        modules="numpy",
    )
    return x_force_fn(
        get_squeeze_ratio_value(eta_param),
        eta_param.eta_m,
        eta_param.eta_lambda,
        eta_param.eta_omega,
    )


def get_friction_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    # d p_like / dt = -stiffness * x_like - friction * p_like
    p_force_expr = get_transformed_deterministic_derivative("p_like").subs(
        {alpha_prime: 1j, sp.Symbol("V_1"): 0, KBT: KBT_value, hbar: KBT_value}
    )
    friction_fn = sp.lambdify(
        (squeeze_ratio, eta_m, eta_lambda, eta_omega),
        -p_force_expr,
        modules="numpy",
    )
    return friction_fn(
        get_squeeze_ratio_value(eta_param),
        eta_param.eta_m,
        eta_param.eta_lambda,
        eta_param.eta_omega,
    )


def get_x_stochastic_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    # d x_like / dt = b11 * re(noise)
    x_stochastic_expr = get_transformed_stochastic_derivative("x_like").subs(
        {noise: 1, KBT: KBT_value, hbar: KBT_value}
    )
    x_stochastic_fn = sp.lambdify(
        (squeeze_ratio, eta_m, eta_lambda, eta_omega),
        x_stochastic_expr,
        modules="numpy",
    )
    return x_stochastic_fn(
        get_squeeze_ratio_value(eta_param),
        eta_param.eta_m,
        eta_param.eta_lambda,
        eta_param.eta_omega,
    )


def get_p_stochastic_value(
    eta_param: EtaParameters = eta_params,
) -> np.complex128:
    # d p_like / dt = b22 * im(noise)
    p_stochastic_expr = get_transformed_stochastic_derivative("p_like").subs(
        {noise: 1j, KBT: KBT_value, hbar: KBT_value}
    )
    p_stochastic_fn = sp.lambdify(
        (squeeze_ratio, eta_m, eta_lambda, eta_omega),
        p_stochastic_expr,
        modules="numpy",
    )
    return p_stochastic_fn(
        get_squeeze_ratio_value(eta_param),
        eta_param.eta_m,
        eta_param.eta_lambda,
        eta_param.eta_omega,
    )


def probability_value(
    gaussian_solution,
    x_mesh: np.ndarray,
    p_mesh: np.ndarray,
    eta_param: EtaParameters = eta_params,
) -> np.ndarray:
    """Calculate the probability value from the Gaussian ansatz solution."""
    x_force_value = get_x_force_value(eta_param)
    friction_value = get_friction_value(eta_param)
    inertia_value = get_inertia_value(eta_param)
    stiffness_value = get_stiffness_value(eta_param)
    x_stochastic_value = get_x_stochastic_value(eta_param)
    p_stochastic_value = get_p_stochastic_value(eta_param)

    s11_expr = gaussian_solution[0][s11].subs(
        {
            X_force: x_force_value,
            friction: friction_value,
            inertia: inertia_value,
            stiffness: stiffness_value,
            b11: x_stochastic_value,
            b22: p_stochastic_value,
        }
    )
    s12_expr = gaussian_solution[0][s12].subs(
        {
            X_force: x_force_value,
            friction: friction_value,
            inertia: inertia_value,
            stiffness: stiffness_value,
            b11: x_stochastic_value,
            b22: p_stochastic_value,
        }
    )
    s22_expr = gaussian_solution[0][s22].subs(
        {
            X_force: x_force_value,
            friction: friction_value,
            inertia: inertia_value,
            stiffness: stiffness_value,
            b11: x_stochastic_value,
            b22: p_stochastic_value,
        }
    )
    z = sp.Matrix([x, p])
    sigma = sp.Matrix([[s11_expr, s12_expr], [s12_expr, s22_expr]])
    normalization = 2 * np.pi / sp.sqrt(sp.det(sigma))
    f_ss = sp.exp(-0.5 * (z.T * sigma * z)[0]) / normalization  # type: ignore unknown
    f_ss_fn = sp.lambdify((x, p), f_ss, modules="numpy")
    return f_ss_fn(x_mesh, p_mesh)  # type: ignore unknown


def _get_energy_expression() -> Path:
    return Path(".cache/get_energy_expression")


@file_cached(_get_energy_expression)
@timed
def get_energy_expression() -> sp.Expr:
    """Get the energy expression in terms of alpha_prime."""
    energy = get_kinetic_term() + get_harmonic_term()
    energy = formula_from_expr(energy)
    energy = expectation_from_formula(energy)
    energy = squeeze_ratio_from_zeta_expr(energy)
    energy = dimensionless_from_full(energy)
    return alpha_prime_from_alpha(energy)


def energy_value(
    x_mesh: np.ndarray,
    p_mesh: np.ndarray,
    eta_param: EtaParameters = eta_params,
) -> np.ndarray:
    energy = get_energy_expression()
    r_eq = evaluate_equilibrium_squeeze_ratio(
        eta_lambda=eta_param.eta_lambda,
        eta_omega=eta_param.eta_omega,
        eta_m=eta_param.eta_m,
    )
    energy = energy.subs(
        {
            eta_lambda: eta_param.eta_lambda,
            eta_omega: eta_param.eta_omega,
            eta_m: eta_param.eta_m,
            squeeze_ratio: r_eq,
            KBT: KBT_value,
            hbar: hbar_value,
        }
    )
    energy_fn = sp.lambdify(
        (sp.re(alpha_prime), sp.im(alpha_prime)), energy, modules="numpy"
    )
    energy_vals = energy_fn(x_mesh, p_mesh)
    assert np.isclose(energy_vals.imag, np.zeros_like(energy_vals)).all(), (
        "Energy values should be real."
    )
    return np.real(energy_vals)


def project_energy(
    probability: np.ndarray,
    x_mesh: np.ndarray,
    p_mesh: np.ndarray,
    E_min=None,
    E_max=None,
    num_E=100,
):
    """
    Parameters
    ----------
    probability : ndarray
        Probability density P(x,p) evaluated on a grid.
    x_mesh : ndarray
        Meshgrid of x values.
    p_mesh : ndarray
        Meshgrid of p values.
    E_min : float, optional
        Minimum energy value for histogram binning. If None, use min(energy).
    E_max : float, optional
        Maximum energy value for histogram binning. If None, use max(energy).
    num_E : int, optional
        Number of energy bins for histogramming.

    Returns
    -------
    E_centers : ndarray
        Centers of energy bins.
    P_E : ndarray
        Probability density of energy.
    """
    dx = x_mesh[0, 1] - x_mesh[0, 0]
    dp = p_mesh[1, 0] - p_mesh[0, 0]
    energy_vals = energy_value(x_mesh, p_mesh)

    # Flatten
    prob_flat = probability.ravel() * dx * dp
    energy_flat = energy_vals.ravel()

    # Energy bin edges
    if E_min is None:
        E_min = energy_flat.min()
    if E_max is None:
        E_max = energy_flat.max()
    E_bins = np.linspace(E_min, E_max, num_E + 1)
    dE = E_bins[1] - E_bins[0]

    # Histogram with probability weights
    P_E, _ = np.histogram(energy_flat, bins=E_bins, weights=prob_flat)

    # Normalize so ∫ P(E) dE = 1
    P_E /= np.sum(P_E) * dE

    E_centers = 0.5 * (E_bins[:-1] + E_bins[1:])
    return E_centers, P_E


def plot_probability_distribution(
    gaussian_solution,
    x_mesh: np.ndarray,
    p_mesh: np.ndarray,
    eta_param: EtaParameters = eta_params,
    *,
    measure: plot.Measure = "real",
) -> tuple[Figure, Axes, QuadMesh]:
    """Construct the probability distribution function from the Gaussian ansatz solution."""
    f_ss = probability_value(
        gaussian_solution, x_mesh=x_mesh, p_mesh=p_mesh, eta_param=eta_param
    )
    fig, ax = plot.get_figure()
    mesh = ax.pcolormesh(
        x_mesh[0, :],
        p_mesh[:, 0],
        plot.get_measured_data(f_ss, measure),  # type: ignore args
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    return fig, ax, mesh


if __name__ == "__main__":
    x_mesh = np.linspace(-2e-10, 2e-10, 1000)
    p_mesh = np.linspace(-2e-10, 2e-10, 1000)
    x_mesh, p_mesh = np.meshgrid(x_mesh, p_mesh)
    energy = energy_value(x_mesh, p_mesh)

    fp_eq_diagonalized = derive_fokker_planck([x, p], drift, B).subs({b12: 0, b21: 0})
    gaussian_solutions = gaussian_ansatz(fp_eq_diagonalized, diffusion="diagonal")
    probability_val = probability_value(
        gaussian_solutions, x_mesh=x_mesh, p_mesh=p_mesh
    )
    sum_of_prob = (
        np.sum(probability_val)
        * (x_mesh[0, 1] - x_mesh[0, 0])
        * (p_mesh[1, 0] - p_mesh[0, 0])
    )
    assert np.isclose(sum_of_prob, 1.0), (
        f"Probability normalization failed: {sum_of_prob}"
    )

    # sort the energy values and probability values
    sorted_indices = np.argsort(energy.ravel())
    energy_val_sorted = energy.ravel()[sorted_indices]
    probability_val_sorted = probability_val.ravel()[sorted_indices]
    fig, ax = plot.get_figure()
    ax.scatter(
        energy,
        np.log(probability_val),
        label="Energy vs Probability Density",
    )
    # plot a straight line with slope 1/KBT_value, starting from the minimum energy value
    ax.scatter(
        energy,
        -energy / (KBT_value) + np.log(probability_val_sorted[0]),
        label="1/KBT Line",
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Log Probability Density")
    ax.set_title("Energy vs Probability Density")
    ax.legend(loc="upper right")
    fig.savefig("energy_vs_probability.png", dpi=300)
    fig.clear()

    fig, ax, mesh = plot_probability_distribution(
        gaussian_solutions, x_mesh=x_mesh, p_mesh=p_mesh
    )
    mesh.set_clim(0, None)
    ax.set_title("Probability Distribution Function")
    fig.savefig("probability_distribution.png", dpi=300)
    fig.clear()
