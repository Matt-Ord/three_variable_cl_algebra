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


def get_low_friction_equilibrium_derivative(
    ty: Literal["zeta", "alpha", "phi"],
) -> sp.Expr:
    """Get the classical equilibrium derivative for the system."""
    equilibrium_zeta = sp.limit(get_equilibrium_zeta(positive=True), eta_lambda, sp.oo)

    deterministic = get_deterministic_derivative(ty).subs({zeta: equilibrium_zeta})
    deterministic = sp.limit(deterministic, eta_lambda, sp.oo)

    stochastic = get_stochastic_derivative(ty).subs({zeta: equilibrium_zeta})
    # Take the power to leading order in 1 / eta_lambda, since at infinity
    # the stochastic term is zero!
    stochastic = sp.simplify(
        stochastic.subs(eta_lambda, 1 / eta_lambda)
        .as_leading_term(eta_lambda)
        .subs(eta_lambda, 1 / eta_lambda)  # type: ignore unknown
    )
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

# Now calculate the x and p derivatives at low friction
low_friction_zeta = sp.limit(get_equilibrium_zeta(positive=True), eta_lambda, sp.oo)

# Split the alpha derivative into real and imaginary parts, and replace
# alpha with x and p expressions.
alpha_derivative_in_xp = xp_expression_from_alpha(low_friction_alpha_derivative).subs(
    zeta, low_friction_zeta
)
re_alpha_derivative = sp.simplify(sp.re(alpha_derivative_in_xp))
im_alpha_derivative = sp.simplify(sp.im(alpha_derivative_in_xp))

# Write <x> and <p> in terms of the real and imaginary parts of alpha.
# Then replace re(alpha) and im(alpha) with their derivatives.
# This gives us d/dt <x> etc, since zeta is constant at low friction.
low_friction_x_derivative = sp.simplify(
    expect_x.subs({zeta: low_friction_zeta, alpha: sp.re(alpha) + 1j * sp.im(alpha)}),
    rational=True,
).subs({sp.re(alpha): re_alpha_derivative, sp.im(alpha): im_alpha_derivative})
sp.print_latex(
    sp.Eq(  # type: ignore unknown
        sp.Symbol(r"\frac{dx}{dt}"),
        low_friction_x_derivative,
    )
)

low_friction_p_derivative = sp.simplify(
    expect_p.subs({zeta: low_friction_zeta, alpha: sp.re(alpha) + 1j * sp.im(alpha)}),
    rational=True,
).subs({sp.re(alpha): re_alpha_derivative, sp.im(alpha): im_alpha_derivative})
sp.print_latex(
    sp.Eq(  # type: ignore unknown
        sp.Symbol(r"\frac{dp}{dt}"),
        sp.expand(sp.simplify(low_friction_p_derivative)),
    )
)
