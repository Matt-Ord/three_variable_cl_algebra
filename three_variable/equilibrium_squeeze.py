# Dimensionless Parameters for the squeezing operator analysis
from __future__ import annotations

from functools import cache

import numpy as np
import sympy as sp
from scipy.constants import hbar as hbar_value  # type: ignore scipy
from slate.util import timed
from sympy.physics.units import hbar

from three_variable.coherent_states import expect_x_squared, uncertainty_squared
from three_variable.projected_sse import (
    get_environment_derivative,
    get_system_derivative,
)
from three_variable.symbols import eta_lambda, eta_m, eta_omega, zeta

squeeze_ratio = sp.Symbol(r"R")
ratio_expr = (1 - zeta) / (1 + zeta)


def squeeze_ratio_from_zeta_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing parameter to the squeezing ratio."""
    zeta_expr = sp.solve(ratio_expr - squeeze_ratio, zeta)[0]
    return sp.simplify(expr.subs(zeta, zeta_expr), rational=True)


def zeta_from_squeeze_ratio_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing ratio to the squeezing parameter."""
    return sp.simplify(expr.subs(squeeze_ratio, ratio_expr), rational=True)


@timed
def get_squeeze_derivative() -> sp.Expr:
    """Get the squeeze derivative with respect to zeta."""
    expr_system = get_system_derivative("zeta")
    expr_environment = get_environment_derivative("zeta")

    expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
    expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))

    return sp.factor_terms(expr_r_system + expr_r_environment, fraction=True)


@timed
@cache
def get_equilibrium_squeeze_ratio(*, positive: bool = True) -> sp.Expr:
    factored = get_squeeze_derivative()
    numer, _denom = sp.together(factored).as_numer_denom()
    return sp.solve(numer, squeeze_ratio)[0 if positive else 1]


_eta_m_symbol = eta_m
_eta_omega_symbol = eta_omega
_eta_lambda_symbol = eta_lambda


def evaluate_equilibrium_squeeze_ratio(
    eta_m: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    *,
    positive: bool = True,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    expression = get_equilibrium_squeeze_ratio(positive=positive)
    expression = expression.subs({hbar: hbar_value})
    expression_lambda = sp.lambdify(
        (_eta_m_symbol, _eta_omega_symbol, _eta_lambda_symbol), expression
    )
    return expression_lambda(eta_m, eta_omega, eta_lambda)  # type: ignore[no-untyped-call]


def evaluate_equilibrium_uncertainty(
    eta_m: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    *,
    positive: bool = True,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complexfloating]]:
    equilibrium_R = evaluate_equilibrium_squeeze_ratio(  # noqa: N806
        eta_m, eta_omega, eta_lambda, positive=positive
    )

    ratio_formula = sp.factor(squeeze_ratio_from_zeta_expr(uncertainty_squared))
    ratio_formula = ratio_formula.subs(
        {
            squeeze_ratio + sp.conjugate(squeeze_ratio): 2 * sp.re(squeeze_ratio),
            squeeze_ratio * sp.conjugate(squeeze_ratio): sp.Abs(squeeze_ratio) ** 2,
            hbar: 1,
        }
    )
    uncertainty_from_ratio = sp.lambdify((squeeze_ratio), ratio_formula)
    return np.real_if_close(uncertainty_from_ratio(equilibrium_R))  # type: ignore[no-untyped-call]


def evaluate_equilibrium_expect_x_squared(
    eta_m: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    *,
    positive: bool = True,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complexfloating]]:
    equilibrium_R = evaluate_equilibrium_squeeze_ratio(  # noqa: N806
        eta_m, eta_omega, eta_lambda, positive=positive
    )
    ratio_formula = sp.factor(squeeze_ratio_from_zeta_expr(expect_x_squared))
    ratio_formula = ratio_formula.subs(
        {
            squeeze_ratio + sp.conjugate(squeeze_ratio): 2 * sp.re(squeeze_ratio),
            squeeze_ratio * sp.conjugate(squeeze_ratio): sp.Abs(squeeze_ratio) ** 2,
            hbar: 1,
        }
    )

    uncertainty_from_ratio = sp.lambdify((squeeze_ratio), ratio_formula)
    return np.real_if_close(uncertainty_from_ratio(equilibrium_R))  # type: ignore[no-untyped-call]
