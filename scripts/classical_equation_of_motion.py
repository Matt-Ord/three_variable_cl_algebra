from __future__ import annotations

from typing import Literal

import sympy as sp
from sympy import Add

from three_variable.equilibrium_squeeze import (
    get_equilibrium_derivative,
    get_equilibrium_zeta,
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
from three_variable.symbols import eta_lambda, eta_omega, p, x, zeta


def get_explicit_equilibrium_derivative(
    params: EtaParameters, *, ty: Literal["zeta", "alpha"]
) -> sp.Expr:
    """Get numerical time derivatives for the alpha, x, and p at equilibrium zeta."""
    return explicit_from_dimensionless(get_equilibrium_derivative(ty), params)


LOW_FRICTION_ZETA = sp.limit(get_equilibrium_zeta(positive=True), eta_lambda, sp.oo)


def get_low_temperature_equilibrium_value(expr: sp.Expr) -> sp.Expr:
    """Get the equilibrium value of an expression at low friction."""
    # Take the power to leading order in 1 / eta_lambda, since at infinity
    # the stochastic term is zero!

    return sp.simplify(
        next(
            expr.subs({zeta: LOW_FRICTION_ZETA}).lseries(  # type: ignore unknown
                (1 - 4 * eta_omega) ** 2 / eta_lambda
            ),
            sp.Number(0),
        )
    )


def get_low_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    deterministic = get_deterministic_derivative(ty)
    deterministic = get_low_temperature_equilibrium_value(deterministic)

    stochastic = get_stochastic_derivative(ty)
    stochastic = get_low_temperature_equilibrium_value(stochastic)
    return deterministic + stochastic


def group_x_p_terms(expr: sp.Expr) -> sp.Expr:
    """Group the x and p terms in an expression."""
    terms = Add.make_args(sp.collect(sp.expand(expr), [x, p]))
    return sum(sp.simplify(term) for term in terms)  # type: ignore unknown


def get_classical_low_friction_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative(ty)
    deterministic = get_low_temperature_equilibrium_value(deterministic)
    deterministic = group_x_p_terms(deterministic)

    stochastic = get_classical_stochastic_derivative(ty)
    stochastic = get_low_temperature_equilibrium_value(stochastic)
    return deterministic + stochastic


if __name__ == "__main__":
    low_friction_alpha_derivative = get_low_friction_equilibrium_derivative("alpha")
    low_friction_alpha_derivative = low_friction_alpha_derivative.subs(
        sp.Symbol("V_1"), 0
    )
    print("alpha derivatives in Classical Limit:")
    sp.print_latex(
        sp.Eq(  # type: ignore unknown
            sp.Symbol(r"\frac{d\alpha}{dt}"),
            low_friction_alpha_derivative,
        )
    )
    print()

    x_derivative = get_classical_stochastic_derivative("x")
    x_derivative = x_derivative.subs(sp.Symbol("V_1"), 0)
    sp.print_latex(
        sp.Eq(  # type: ignore unknown
            sp.Symbol(r"\frac{dx}{dt}"),
            x_derivative,
        )
    )
    print()
    # Now we calculate the x and p derivatives at low friction
    low_friction_x_derivative = get_classical_low_friction_equilibrium_derivative("x")
    low_friction_x_derivative = low_friction_x_derivative.subs(sp.Symbol("V_1"), 0)
    sp.print_latex(
        sp.Eq(  # type: ignore unknown
            sp.Symbol(r"\frac{dx}{dt}"),
            low_friction_x_derivative,
        )
    )
    print()

    p_derivative = get_classical_stochastic_derivative("p")
    p_derivative = p_derivative.subs(sp.Symbol("V_1"), 0)
    sp.print_latex(
        sp.Eq(  # type: ignore unknown
            sp.Symbol(r"\frac{dp}{dt}"),
            p_derivative,
        )
    )
    print()
    low_friction_p_derivative = get_classical_low_friction_equilibrium_derivative("p")
    low_friction_p_derivative = low_friction_p_derivative.subs(sp.Symbol("V_1"), 0)
    sp.print_latex(
        sp.Eq(  # type: ignore unknown
            sp.Symbol(r"\frac{dp}{dt}"),
            low_friction_p_derivative,
        )
    )
