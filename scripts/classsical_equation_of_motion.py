from __future__ import annotations

from typing import Literal

import sympy as sp

from three_variable.coherent_states import expect_p, expect_x, xp_expression_from_alpha
from three_variable.equilibrium_squeeze import (
    get_equilibrium_derivative,
    get_equilibrium_zeta,
)
from three_variable.projected_sse import (
    get_deterministic_derivative,
    get_stochastic_derivative,
)
from three_variable.simulation import (
    EtaParameters,
    explicit_from_dimensionless,
)
from three_variable.symbols import alpha, eta_lambda, zeta


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


LOW_FRICTION_ZETA = sp.limit(get_equilibrium_zeta(positive=True), eta_lambda, sp.oo)


def get_low_temperature_equilibrium_value(expr: sp.Expr) -> sp.Expr:
    """Get the equilibrium value of an expression at low friction."""
    # Take the power to leading order in 1 / eta_lambda, since at infinity
    # the stochastic term is zero!
    temp_symbol = sp.Symbol("inverse_eta_lambda", positive=True, real=True)

    return sp.simplify(
        next(
            expr.subs({zeta: LOW_FRICTION_ZETA})
            .subs(eta_lambda, 1 / temp_symbol)  # type: ignore unknown
            .lseries(temp_symbol),
            sp.Number(0),
        ).subs(temp_symbol, 1 / eta_lambda)
    )


def get_low_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    deterministic = get_deterministic_derivative(ty)
    deterministic = get_low_temperature_equilibrium_value(deterministic)

    stochastic = get_stochastic_derivative(ty)
    stochastic = get_low_temperature_equilibrium_value(stochastic)
    return deterministic + stochastic


low_friction_alpha_derivative = get_low_friction_equilibrium_derivative("alpha")
low_friction_alpha_derivative = low_friction_alpha_derivative.subs(sp.Symbol("V_1"), 0)
print("alpha derivatives in Classical Limit:")
sp.print_latex(
    sp.Eq(  # type: ignore unknown
        sp.Symbol(r"\frac{d\alpha}{dt}"),
        low_friction_alpha_derivative,
    )
)


def get_classical_low_friction_equilibrium_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    deterministic = get_classical_deterministic_derivative(ty)
    deterministic = get_low_temperature_equilibrium_value(deterministic)

    stochastic = get_classical_stochastic_derivative(ty)
    stochastic = get_low_temperature_equilibrium_value(stochastic)
    return deterministic + stochastic


# Now we calculate the x and p derivatives at low friction
low_friction_x_derivative = get_classical_low_friction_equilibrium_derivative("x")
low_friction_x_derivative = low_friction_x_derivative.subs(sp.Symbol("V_1"), 0)
sp.print_latex(
    sp.Eq(  # type: ignore unknown
        sp.Symbol(r"\frac{dx}{dt}"),
        low_friction_x_derivative,
    )
)

low_friction_p_derivative = get_classical_low_friction_equilibrium_derivative("p")
low_friction_p_derivative = low_friction_p_derivative.subs(sp.Symbol("V_1"), 0)
sp.print_latex(
    sp.Eq(  # type: ignore unknown
        sp.Symbol(r"\frac{dp}{dt}"),
        sp.expand(sp.simplify(low_friction_p_derivative)),
    )
)
