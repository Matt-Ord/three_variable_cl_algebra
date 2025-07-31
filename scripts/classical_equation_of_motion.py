from __future__ import annotations

import itertools
from functools import cache
from typing import Literal

import sympy as sp
from sympy import Add

from three_variable.coherent_states import expect_p, expect_x, xp_expression_from_alpha
from three_variable.equilibrium_squeeze import (
    get_equilibrium_derivative,
    get_equilibrium_squeeze_ratio,
    get_equilibrium_zeta,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import (
    get_deterministic_derivative,
    get_stochastic_derivative,
)
from three_variable.simulation import (
    EtaParameters,
    explicit_from_dimensionless,
)
from three_variable.symbols import (
    alpha,
    eta_lambda,
    eta_m,
    noise,
    p,
    x,
    zeta,
)

alpha_coeff_alpha = sp.Symbol("alpha_coeff_alpha", complex=True)
alpha_coeff_alpha_bar = sp.Symbol("alpha_coeff_alpha_bar", complex=True)
r0 = sp.Symbol("R_0", complex=True)  # R = eta_m * r0


def get_explicit_equilibrium_derivative(
    params: EtaParameters, *, ty: Literal["zeta", "alpha"]
) -> sp.Expr:
    """Get numerical time derivatives for the alpha, x, and p at equilibrium zeta."""
    return explicit_from_dimensionless(get_equilibrium_derivative(ty), params)


def get_classical_deterministic_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    # Write <x> and <p> in terms of the real and imaginary parts of alpha.
    # Then replace re(alpha) and im(alpha) with their derivatives.
    # This gives us d/dt <x> etc.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative_xp = xp_expression_from_alpha(get_deterministic_derivative("alpha"))

    expect_var = expect_x if ty == "x" else expect_p
    expect_var = expect_var.subs({alpha: sp.re(alpha) + 1j * sp.im(alpha)})

    return sp.simplify(
        expect_var.subs(
            {sp.re(alpha): sp.re(derivative_xp), sp.im(alpha): sp.im(derivative_xp)}
        ),
        rational=True,
    )


def get_classical_deterministic_derivative_complex_conj_pair(
    ty: Literal["x", "p"],
) -> sp.Expr:
    # Write <x> and <p> in terms of the real and imaginary parts of alpha.
    # Then replace re(alpha) and im(alpha) with their derivatives.
    # This gives us d/dt <x> etc.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative_xp = xp_expression_from_alpha(get_deterministic_derivative("alpha"))

    expect_var = expect_x if ty == "x" else expect_p

    return sp.simplify(
        expect_var.subs(
            {alpha: derivative_xp, sp.conjugate(alpha): sp.conjugate(derivative_xp)}
        ),
        rational=True,
    )


def get_classical_stochastic_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    # Write <x> and <p> in terms of the real and imaginary parts of alpha.
    # Then replace re(alpha) and im(alpha) with their derivatives.
    # This gives us d/dt <x> etc.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative_xp = xp_expression_from_alpha(get_stochastic_derivative("alpha"))

    expect_var = expect_x if ty == "x" else expect_p
    expect_var = expect_var.subs({alpha: sp.re(alpha) + 1j * sp.im(alpha)})

    return sp.simplify(
        expect_var.subs(
            {sp.re(alpha): sp.re(derivative_xp), sp.im(alpha): sp.im(derivative_xp)}
        ),
        rational=True,
    )


def get_classical_stochastic_derivative_complex_conj_pair(
    ty: Literal["x", "p"],
) -> sp.Expr:
    # Write <x> and <p> in terms of the real and imaginary parts of alpha.
    # Then replace re(alpha) and im(alpha) with their derivatives.
    # This gives us d/dt <x> etc.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative_xp = xp_expression_from_alpha(get_stochastic_derivative("alpha"))

    expect_var = expect_x if ty == "x" else expect_p

    return sp.simplify(
        expect_var.subs(
            {alpha: derivative_xp, sp.conjugate(alpha): sp.conjugate(derivative_xp)}
        ),
        rational=True,
    )


# LOW_FRICTION_ZETA = sp.limit(get_equilibrium_zeta(positive=True), eta_lambda, sp.oo)
low_friction = get_equilibrium_zeta(positive=True).lseries(eta_lambda, sp.oo)  # type: ignore sp
LOW_FRICTION_ZETA = sum(sp.simplify(e) for e in itertools.islice(low_friction, 3))


def get_low_friction_equilibrium_value(expr: sp.Expr) -> sp.Expr:
    """Get the equilibrium value of an expression at low friction."""
    # Take the power to leading order in 1 / eta_lambda, since at infinity
    # the stochastic term is zero!
    low_friction = expr.subs({zeta: LOW_FRICTION_ZETA}).lseries(  # type: ignore unknown
        eta_lambda, sp.oo
    )
    return sum(sp.simplify(e) for e in itertools.islice(low_friction, 4))  # type: ignore sp
    # return sp.simplify(
    #     next(
    #         expr.subs({zeta: LOW_FRICTION_ZETA}).lseries(  # type: ignore unknown
    #             (1 - 4 * eta_omega) ** 2 / eta_lambda
    #         ),
    #         sp.Number(0),
    #     )
    # )


def get_low_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    deterministic = get_deterministic_derivative(ty)
    deterministic = get_low_friction_equilibrium_value(deterministic)

    stochastic = 0  # get_stochastic_derivative(ty)
    stochastic = 0  # get_low_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


def get_high_friction_equilibrium_value(expr: sp.Expr) -> sp.Expr:
    """Get the equilibrium value of an expression at low friction."""
    # Return series in eta_lambda, since at high friction
    zeta_eq = get_equilibrium_zeta(positive=True)
    zeta_eq = sp.series(zeta_eq, eta_lambda, 0, n=1).removeO()
    expr = expr.subs({zeta: sp.simplify(zeta_eq)})
    # series = sp.series(expr, eta_lambda, 0, n=3).removeO()
    return sp.simplify(expr)
    # return sp.simplify(series).removeO()


def get_high_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    deterministic = get_deterministic_derivative(ty)
    deterministic = get_high_friction_equilibrium_value(deterministic)

    stochastic = get_stochastic_derivative(ty)
    stochastic = get_high_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


def get_high_mass_equilibrium_value(expr: sp.Expr, expanding: bool = True) -> sp.Expr:
    """Get the equilibrium value of an expression at high mass, in power series of 1/eta_m."""
    # Expand to leading order in eta_m
    expr = expr.subs({squeeze_ratio: eta_m * r0})
    if not expanding:
        return sp.simplify(expr)
    high_mass_series = expr.lseries(  # type: ignore unknown
        eta_m, sp.oo
    )
    return sum(sp.simplify(e) for e in itertools.islice(high_mass_series, 2))  # type: ignore sp


def substitute_back_r0(expr: sp.Expr) -> sp.Expr:
    """Substitute back r0 in an expression."""
    # Substitute r0 with the squeeze ratio
    equilibrium_r0 = get_equilibrium_squeeze_ratio() / eta_m
    expr = expr.subs({r0: equilibrium_r0})
    return sp.simplify(expr)


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

    stochastic = 0  # get_classical_stochastic_derivative(ty)
    stochastic = 0  # get_low_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


def get_classical_high_friction_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative_complex_conj_pair(ty)
    deterministic = squeeze_ratio_from_zeta_expr(deterministic)
    deterministic = group_x_p_terms(deterministic)
    deterministic = get_high_friction_equilibrium_value(deterministic)

    stochastic = get_classical_stochastic_derivative(ty)
    stochastic = get_high_friction_equilibrium_value(stochastic)
    return deterministic + stochastic


@cache
def get_classical_equilibrium_derivative_ratio(
    ty: Literal["x", "p"],
    part: Literal["deterministic", "stochastic", "total"] = "total",
) -> sp.Expr:
    if part == "deterministic":
        deterministic = get_classical_deterministic_derivative_complex_conj_pair(ty)
        deterministic = squeeze_ratio_from_zeta_expr(deterministic)
        return group_x_p_terms(deterministic)

    if part == "stochastic":
        stochastic = get_classical_stochastic_derivative_complex_conj_pair(ty)
        return squeeze_ratio_from_zeta_expr(stochastic)

    deterministic = get_classical_deterministic_derivative_complex_conj_pair(ty)
    deterministic = squeeze_ratio_from_zeta_expr(deterministic)
    deterministic = group_x_p_terms(deterministic)
    stochastic = get_classical_stochastic_derivative_complex_conj_pair(ty)
    stochastic = squeeze_ratio_from_zeta_expr(stochastic)
    return deterministic + stochastic


@cache
def get_classical_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative_complex_conj_pair(ty)
    deterministic = group_x_p_terms(deterministic)

    stochastic = get_classical_stochastic_derivative_complex_conj_pair(ty)
    return deterministic + stochastic


if __name__ == "__main__":
    low_friction_x_derivative = get_classical_equilibrium_derivative_ratio("x")
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
        sp.simplify(e) for e in itertools.islice(x_dependence_x_low_friction, 2)
    )  # type: ignore sp
    print("x dependence on x:")
    sp.print_latex(x_dependence_x_low_friction_series)
