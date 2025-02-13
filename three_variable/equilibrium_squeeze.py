# Dimensionless Parameters for the squeezing operator analysis
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import sympy as sp

from .decorators import timed
from .projected_sse import get_squeeze_derivative_beta
from .symbols import beta, eta_lambda, eta_m, eta_omega, mu, r, theta

if TYPE_CHECKING:
    import numpy as np

# We actually solve for R not for beta
R = sp.Symbol("R")
R_from_beta = (1 - sp.conjugate(beta)) / (1 + sp.conjugate(beta))


# squeeze_derivative_lambda = (
#     (1 - sp.conjugate(beta)) ** 2
#     - 4 * eta_m * (1 - sp.conjugate(beta)) * (1 + sp.conjugate(beta))
#     - 4 * eta_m**2 * (1 + sp.conjugate(beta)) ** 2
# )
# squeeze_derivative_mass = (1 - sp.conjugate(beta)) ** 2 - (
#     (eta_m / eta_omega) ** 2 * (1 + sp.conjugate(beta)) ** 2
# )
# squeeze_derivative = (
#     sp.I * squeeze_derivative_mass + (1 / (4 * eta_lambda)) * squeeze_derivative_lambda
# )


@cache
@timed
def get_squeeze_derivative_R() -> sp.Expr:
    squeeze_derivative = get_squeeze_derivative_beta()
    beta_from_R = sp.solve(R_from_beta - R, sp.conjugate(beta))[0]
    subbed = squeeze_derivative.subs({sp.conjugate(beta): beta_from_R})
    mu_squared_from_beta = 1 / (1 - beta * sp.conjugate(beta))
    mu_from_r = mu_squared_from_beta.subs(
        {sp.conjugate(beta): beta_from_R, beta: sp.conjugate(beta_from_R)}
    )
    numer, denom = sp.together(mu_from_r).as_numer_denom()
    mu_from_r = sp.simplify(numer) / sp.factor(sp.simplify(denom))
    return sp.together(sp.simplify(subbed)).subs({mu**2: mu_from_r})


@cache
def get_equilibrium_squeeze_R(*, positive: bool = False) -> sp.Expr:
    squeeze_derivative = get_squeeze_derivative_R()
    return sp.solve(squeeze_derivative, R)[0 if positive else 1]


_eta_m_symbol = eta_m
_eta_omega_symbol = eta_omega
_eta_lambda_symbol = eta_lambda


def evaluate_equilibrium_R(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    expression = get_equilibrium_squeeze_R(positive=positive)
    expression_lambda = sp.lambdify(
        (_eta_m_symbol, _eta_omega_symbol, _eta_lambda_symbol), expression
    )
    return expression_lambda(eta_m, eta_omega, eta_lambda)


@cache
def get_equilibrium_squeeze_beta(*, positive: bool = False) -> sp.Expr:
    beta_from_R = sp.solve(R_from_beta - R, sp.conjugate(beta))[0]
    equilibrium_R = get_equilibrium_squeeze_R(positive=positive)  # noqa: N806
    beta_expression = sp.conjugate(beta_from_R.subs({R: equilibrium_R}))
    return sp.simplify(beta_expression)


def evaluate_equilibrium_beta(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    expression = get_equilibrium_squeeze_beta(positive=positive)
    expression_lambda = sp.lambdify(
        (_eta_m_symbol, _eta_omega_symbol, _eta_lambda_symbol), expression
    )
    return expression_lambda(eta_m, eta_omega, eta_lambda)


@cache
def get_equilibrium_squeeze_derivative() -> sp.Expr:
    squeeze_derivative = get_squeeze_derivative_beta()
    subbed = squeeze_derivative.subs({beta: get_equilibrium_squeeze_beta()})
    return sp.simplify(subbed)


@cache
def get_squeeze_derivative_gradient_beta() -> sp.Expr:
    temp = sp.Symbol("temp")
    squeeze_derivative = get_squeeze_derivative_beta()
    gradient = sp.diff(squeeze_derivative.subs({sp.conjugate(beta): temp}), temp).subs(
        {temp: sp.conjugate(beta)}
    )
    return sp.simplify(gradient)


@cache
def get_equilibrium_squeeze_derivative_gradient(*, positive: bool = False) -> sp.Expr:
    gradient = get_squeeze_derivative_gradient_beta()
    gradient = gradient.subs({beta: get_equilibrium_squeeze_beta(positive=positive)})
    gradient = sp.together(gradient)
    neumerator, denominator = gradient.as_numer_denom()
    gradient = sp.simplify(neumerator) / sp.simplify(denominator)
    return sp.simplify(gradient)


uncertainty_x = (
    sp.Pow(sp.cosh(r), 2)
    + sp.Pow(sp.sinh(r), 2)
    - 2 * sp.cosh(r) * sp.sinh(r) * sp.cos(theta)
) / 4
uncertainty_p = (
    sp.Pow(sp.cosh(r), 2)
    + sp.Pow(sp.sinh(r), 2)
    + 2 * sp.cosh(r) * sp.sinh(r) * sp.cos(theta)
) / 4
uncertainty = uncertainty_x * uncertainty_p


@cache
def get_uncertainty_x_beta() -> sp.Expr:
    r_from_beta = sp.atanh(sp.Abs(beta))
    theta_from_beta = sp.arg(beta)
    subbed = uncertainty_x.subs({r: r_from_beta, theta: theta_from_beta})
    subbed = subbed.subs(
        {sp.cos(sp.arg(beta)): (beta + sp.conjugate(beta)) / (2 * sp.Abs(beta))}
    )
    return sp.simplify(subbed)


@cache
def get_uncertainty_p_beta() -> sp.Expr:
    r_from_beta = sp.atanh(sp.Abs(beta))
    theta_from_beta = sp.arg(beta)
    subbed = uncertainty_p.subs({r: r_from_beta, theta: theta_from_beta})
    subbed = subbed.subs(
        {sp.cos(sp.arg(beta)): (beta + sp.conjugate(beta)) / (2 * sp.Abs(beta))}
    )
    return sp.simplify(subbed)


@cache
def get_uncertainty_beta() -> sp.Expr:
    uncertainty_x_beta = get_uncertainty_x_beta()
    uncertainty_p_beta = get_uncertainty_p_beta()
    uncertainty = uncertainty_x_beta * uncertainty_p_beta
    uncertainty = uncertainty.subs({sp.Abs(beta) ** 2: sp.conjugate(beta) * beta})
    numerator, denom = sp.together(uncertainty).as_numer_denom()
    return sp.factor(sp.simplify(sp.expand(numerator))) / sp.factor(sp.simplify(denom))


@cache
def get_uncertainty_R() -> sp.Expr:
    beta_from_R = sp.solve(R_from_beta - R, beta)[0]
    uncertainty_beta = get_uncertainty_beta()
    uncertainty_R = uncertainty_beta.subs({beta: beta_from_R})
    uncertainty_R = sp.together(sp.simplify(sp.expand(uncertainty_R)))
    neum, denom = uncertainty_R.as_numer_denom()
    neum = sp.simplify(neum)
    return neum / sp.factor(denom)


@cache
def get_uncertainty_x_R() -> sp.Expr:
    beta_from_R = sp.solve(R_from_beta - R, beta)[0]
    uncertainty_beta = get_uncertainty_x_beta()
    uncertainty_R = uncertainty_beta.subs({beta: beta_from_R})
    uncertainty_R = sp.together(sp.simplify(sp.expand(uncertainty_R)))
    neum, denom = uncertainty_R.as_numer_denom()
    uncertainty_R = sp.simplify(neum) / sp.factor(denom)
    return uncertainty_R.subs(
        {R * sp.conjugate(R): sp.Abs(R) ** 2, sp.conjugate(R): 2 * sp.re(R) - R}
    )


@cache
def get_uncertainty_p_R() -> sp.Expr:
    beta_from_R = sp.solve(R_from_beta - R, beta)[0]
    uncertainty_beta = get_uncertainty_p_beta()
    uncertainty_R = uncertainty_beta.subs({beta: beta_from_R})
    uncertainty_R = sp.together(sp.simplify(sp.expand(uncertainty_R)))
    neum, denom = uncertainty_R.as_numer_denom()
    uncertainty_R = sp.simplify(neum) / sp.factor(denom)
    return uncertainty_R.subs(
        {R * sp.conjugate(R): sp.Abs(R) ** 2, sp.conjugate(R): 2 * sp.re(R) - R}
    )
