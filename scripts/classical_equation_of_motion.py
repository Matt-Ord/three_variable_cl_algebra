from __future__ import annotations

import itertools
from functools import cache
from typing import Literal

import sympy as sp
from sympy import Add

from three_variable.equilibrium_squeeze import (
    get_classical_derivative_squeeze_ratio,
    get_equilibrium_derivative,
    get_equilibrium_squeeze_ratio,
    get_equilibrium_zeta,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import (
    get_classical_deterministic_derivative,
    get_classical_stochastic_derivative,
    get_deterministic_derivative,
    get_stochastic_derivative,
)
from three_variable.simulation import (
    EtaParameters,
    explicit_from_dimensionless,
)
from three_variable.symbols import (
    eta_lambda,
    noise,
    p,
    x,
    zeta,
)


def get_explicit_equilibrium_derivative(
    params: EtaParameters, *, ty: Literal["zeta", "alpha"]
) -> sp.Expr:
    """Get numerical time derivatives for the alpha, x, and p at equilibrium zeta."""
    return explicit_from_dimensionless(get_equilibrium_derivative(ty), params)


low_friction = get_equilibrium_zeta(positive=True).lseries(eta_lambda, sp.oo)  # type: ignore sp
LOW_FRICTION_ZETA = sum(sp.simplify(e) for e in itertools.islice(low_friction, 3))  # type: ignore sp


def get_low_friction_equilibrium_value(expr: sp.Expr) -> sp.Expr:
    """Get the equilibrium value of an expression at low friction."""
    # Power series expansion in 1 / eta_lambda
    low_friction = expr.subs({zeta: LOW_FRICTION_ZETA}).lseries(  # type: ignore unknown
        eta_lambda, sp.oo
    )
    return sum(sp.simplify(e) for e in itertools.islice(low_friction, 4))  # type: ignore sp


def get_low_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    deterministic = get_deterministic_derivative(ty)
    deterministic = get_low_friction_equilibrium_value(deterministic)

    stochastic = get_stochastic_derivative(ty)
    stochastic = get_low_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


def get_high_friction_equilibrium_value(expr: sp.Expr) -> sp.Expr:
    """Get the equilibrium value of an expression at low friction."""
    # Return series in eta_lambda, since at high friction
    zeta_eq = get_equilibrium_zeta(positive=True)
    zeta_eq = sp.series(zeta_eq, eta_lambda, 0, n=1).removeO()  # type: ignore sp
    expr = expr.subs({zeta: sp.simplify(zeta_eq)})  # type: ignore unknown
    return sp.simplify(expr)


def get_high_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    deterministic = get_deterministic_derivative(ty)
    deterministic = get_high_friction_equilibrium_value(deterministic)

    stochastic = get_stochastic_derivative(ty)
    stochastic = get_high_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


def group_x_p_terms(expr: sp.Expr) -> sp.Expr:
    """Group the x and p terms in an expression."""
    terms = Add.make_args(sp.collect(sp.expand(expr), [x, p]))
    return sum(sp.simplify(term) for term in terms)  # type: ignore unknown


def get_classical_low_friction_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative(ty)
    deterministic = get_low_friction_equilibrium_value(deterministic)
    deterministic = group_x_p_terms(deterministic)

    stochastic = get_classical_stochastic_derivative(ty)
    stochastic = get_low_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


def get_classical_high_friction_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative(ty)
    deterministic = squeeze_ratio_from_zeta_expr(deterministic)
    deterministic = group_x_p_terms(deterministic)
    deterministic = get_high_friction_equilibrium_value(deterministic)

    stochastic = get_classical_stochastic_derivative(ty)
    stochastic = get_high_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


@cache
def get_total_equilibrium_derivative_ratio(
    ty: Literal["x", "p"],
) -> sp.Expr:
    """Get the classical equilibrium derivative for x and p in terms of the squeeze ratio."""
    deterministic = get_classical_derivative_squeeze_ratio(ty, "deterministic")
    stochastic = get_classical_derivative_squeeze_ratio(ty, "stochastic")
    return deterministic + stochastic


@cache
def get_classical_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative(ty)
    deterministic = group_x_p_terms(deterministic)

    stochastic = get_classical_stochastic_derivative(ty)
    return deterministic + stochastic


if __name__ == "__main__":
    low_friction_x_derivative = get_total_equilibrium_derivative_ratio("x")
    low_friction_x_derivative = low_friction_x_derivative.subs(sp.Symbol("V_1"), 0)
    sp.print_latex(
        sp.Eq(  # type: ignore unknown
            sp.Symbol(r"\frac{dx}{dt}"),
            low_friction_x_derivative,
        )
    )
    print()
    x_dependence_x = low_friction_x_derivative.subs(
        {p: 0, x: 1, noise: 0, sp.conjugate(noise): 0}
    )
    equilibrium_ratio = get_equilibrium_squeeze_ratio()
    low_friction = equilibrium_ratio.lseries(eta_lambda, sp.oo)  # type: ignore sp
    low_friction_series = sum(sp.simplify(e) for e in itertools.islice(low_friction, 2))  # type: ignore sp
    print("substituting equilibrium ratio")
    x_dependence_x = sp.simplify(x_dependence_x).subs(
        {squeeze_ratio: low_friction_series}
    )
    x_dependence_x_low_friction = (x_dependence_x).lseries(eta_lambda, sp.oo)  # type: ignore sp
    x_dependence_x_low_friction_series = sum(
        sp.simplify(e)  # type: ignore sp
        for e in itertools.islice(x_dependence_x_low_friction, 2)  # type: ignore sp
    )  # type: ignore sp
    print("x dependence on x:")
    sp.print_latex(x_dependence_x_low_friction_series)  # type: ignore sp
