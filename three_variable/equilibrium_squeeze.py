from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import sympy as sp
from scipy.constants import hbar as hbar_value  # type: ignore scipy
from slate_core.util import timed
from sympy import Add
from sympy.physics.units import hbar

from three_variable.coherent_states import expect_x_squared, uncertainty_squared
from three_variable.projected_sse import (
    get_classical_deterministic_derivative,
    get_classical_stochastic_derivative,
    get_environment_derivative,
    get_full_derivative,
    get_system_derivative,
)
from three_variable.symbols import eta_lambda, eta_m, eta_omega, p, x, zeta
from three_variable.util import file_cached

squeeze_ratio = sp.Symbol(r"R")
ratio_expr = (1 - zeta) / (eta_m * (1 + zeta))


def squeeze_ratio_from_zeta_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing parameter to the squeezing ratio."""
    zeta_expr = sp.solve(ratio_expr - squeeze_ratio, zeta)[0]
    return sp.simplify(expr.subs(zeta, zeta_expr), rational=True)


def zeta_from_squeeze_ratio_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing ratio to the squeezing parameter."""
    return sp.simplify(expr.subs(squeeze_ratio, ratio_expr), rational=True)


def r0_from_squeeze_ratio_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing ratio to the r0 value."""
    return sp.simplify(expr.subs(squeeze_ratio, eta_m * r0))


@timed
def get_squeeze_derivative_zeta() -> sp.Expr:
    """Get the squeeze derivative with respect to zeta."""
    expr_system = get_system_derivative("zeta")
    expr_environment = get_environment_derivative("zeta")

    expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
    expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))

    return sp.factor_terms(expr_r_system + expr_r_environment, fraction=True)


def _get_get_equilibrium_squeeze_ratio_path(*, positive: bool = True) -> Path:
    """Get the path to the cached equilibrium squeeze ratio."""
    return Path(f".cache/equilibrium_squeeze_ratio.{positive}")


@file_cached(_get_get_equilibrium_squeeze_ratio_path)
@timed
def get_equilibrium_squeeze_ratio(*, positive: bool = True) -> sp.Expr:
    factored = get_squeeze_derivative_zeta()
    numer, _denom = sp.together(factored).as_numer_denom()
    return sp.solve(numer, squeeze_ratio)[0 if positive else 1]


def get_equilibrium_zeta(*, positive: bool = True) -> sp.Expr:
    ratio = get_equilibrium_squeeze_ratio(positive=positive)
    zeta_expr = sp.solve(ratio_expr - squeeze_ratio, zeta)[0]
    return sp.simplify(zeta_expr.subs({squeeze_ratio: ratio}))


@timed
@cache
def get_equilibrium_derivative(ty: Literal["zeta", "alpha"]) -> sp.Expr:
    """Get the derivative for the system at equilibrium."""
    if ty == "zeta":
        return sp.Integer(0)
    if ty == "alpha":
        equilibrium_zeta = get_equilibrium_zeta(positive=True)
        return get_full_derivative("alpha").subs({zeta: equilibrium_zeta})
    return None


def get_equilibrium_zeta(*, positive: bool = True) -> sp.Expr:
    ratio = get_equilibrium_squeeze_ratio(positive=positive)
    zeta_expr = sp.solve(ratio_expr - squeeze_ratio, zeta)[0]
    return sp.simplify(zeta_expr.subs({squeeze_ratio: ratio}))


@timed
@cache
def get_equilibrium_derivative(ty: Literal["zeta", "alpha"]) -> sp.Expr:
    """Get the derivative for the system at equilibrium."""
    if ty == "zeta":
        return sp.Integer(0)
    if ty == "alpha":
        equilibrium_zeta = get_equilibrium_zeta(positive=True)
        return get_full_derivative("alpha").subs({zeta: equilibrium_zeta})
    return None


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


def _group_x_p_terms(expr: sp.Expr) -> sp.Expr:
    """Group the x and p terms in an expression."""
    terms = Add.make_args(sp.collect(sp.expand(expr), [x, p]))
    return sum(sp.simplify(term) for term in terms)  # type: ignore unknown


def _get_classical_derivative_squeeze_ratio(
    ty: Literal["x", "p"], part: Literal["deterministic", "stochastic"]
) -> Path:
    """Get the classical equilibrium derivative for x and p in terms of the squeeze ratio."""
    return Path(f".cache/classical_derivative_squeeze_ratio.{ty}.{part}")


@file_cached(_get_classical_derivative_squeeze_ratio)
@timed
def get_classical_derivative_squeeze_ratio(
    ty: Literal["x", "p"], part: Literal["deterministic", "stochastic"]
) -> sp.Expr:
    """Get the classical equilibrium derivative for x and p in terms of the squeeze ratio."""
    if part == "deterministic":
        deterministic = get_classical_deterministic_derivative(ty)
        deterministic = squeeze_ratio_from_zeta_expr(deterministic)
        return _group_x_p_terms(deterministic)

    if part == "stochastic":
        stochastic = get_classical_stochastic_derivative(ty)
        return squeeze_ratio_from_zeta_expr(stochastic)
    return None
