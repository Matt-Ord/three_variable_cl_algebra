from __future__ import annotations

from typing import Literal

import sympy as sp
from sympy.physics.quantum import Dagger

from three_variable.symbols import (
    a_expr,
    alpha,
    formula_from_expr,
    k_0,
    k_minus,
    k_plus,
    p_expr,
    x_expr,
    zeta,
)

d_alpha = sp.Symbol("d_alpha")
d_zeta = sp.Symbol("d_zeta")

DIFFERENTIAL_ACTION_A_DAGGER = d_alpha
DIFFERENTIAL_ACTION_A = alpha + zeta * d_alpha
DIFFERENTIAL_ACTION_K_PLUS = d_zeta
DIFFERENTIAL_ACTION_K_MINUS = (
    0.5 * alpha**2 + 0.5 * zeta + alpha * zeta * d_alpha + zeta**2 * d_zeta
)
DIFFERENTIAL_ACTION_K_0 = 0.25 + 0.5 * alpha * d_alpha + zeta * d_zeta

EXPECTATION_D_ALPHA = (sp.conjugate(alpha) + alpha * sp.conjugate(zeta)) / (
    1 - zeta * sp.conjugate(zeta)
)
EXPECTATION_D_ZETA = ((sp.conjugate(zeta)) / (2 * (1 - zeta * sp.conjugate(zeta)))) + (
    EXPECTATION_D_ALPHA**2 / 2
)


def action_from_formula(expr: sp.Expr) -> sp.Expr:
    return expr.subs(
        {
            k_0: DIFFERENTIAL_ACTION_K_0,
            k_plus: DIFFERENTIAL_ACTION_K_PLUS,
            k_minus: DIFFERENTIAL_ACTION_K_MINUS,
            Dagger(a_expr): DIFFERENTIAL_ACTION_A_DAGGER,
            a_expr: DIFFERENTIAL_ACTION_A,
        }
    )


def action_from_expr(expr: sp.Expr) -> sp.Expr:
    return action_from_formula(formula_from_expr(expr))


def extract_action(expr: sp.Expr, ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    expr = sp.collect(sp.expand(expr), [d_zeta, d_alpha])  # type: ignore[no-untyped-call]
    if ty == "zeta":
        return expr.coeff(d_zeta)
    if ty == "alpha":
        return expr.coeff(d_alpha)
    if ty == "phi":
        return expr.subs({d_zeta: 0, d_alpha: 0})
    msg = f"Unknown type: {ty}"
    raise ValueError(msg)


def expectation_from_action(action: sp.Expr) -> sp.Expr:
    return action.subs(
        {
            d_alpha: EXPECTATION_D_ALPHA,
            d_zeta: EXPECTATION_D_ZETA,
        }
    )


def get_expectation_a() -> sp.Expr:
    """Compute < a >.

    a has the differential action of d / dalpha, so
    < a > = d / d alpha (ln(N)).
    """
    return (alpha + sp.conjugate(alpha) * zeta) / (1 - zeta * sp.conjugate(zeta))


def get_expectation_a_dagger() -> sp.Expr:
    """Compute < a_dagger >.

    a_dagger has the differential action of d / dalpha, so
    < a_dagger > = d / d alpha (ln(N)).
    """
    return EXPECTATION_D_ALPHA


def get_expectation_k_minus() -> sp.Expr:
    """Compute < k_- >.

    K_- has the differential action of d / dzeta, so
    < k_- > = d / dzeta (ln(N)).
    """
    return (zeta / (2 * (1 - zeta * sp.conjugate(zeta)))) + (
        get_expectation_a() ** 2 / 2
    )


def get_expectation_k_plus() -> sp.Expr:
    """Compute < k_+ >.

    K_+ has the differential action of d / dzeta, so
    < k_+ > = d / dzeta (ln(N)).
    """
    return EXPECTATION_D_ZETA


def get_expectation_k_0() -> sp.Expr:
    """Compute < k_+ >.

    K_+ has the differential action of d / dzeta, so
    < k_+ > = d / dzeta (ln(N)).
    """
    return (
        0.25
        + (0.5 * alpha * get_expectation_a_dagger())
        + zeta * get_expectation_k_plus()
    )


def expectation_from_formula(expr: sp.Expr) -> sp.Expr:
    """Get the expectation of an expression which is written in terms of the generators."""
    return expr.subs(
        {
            k_0: get_expectation_k_0(),
            k_plus: get_expectation_k_plus(),
            k_minus: get_expectation_k_minus(),
            Dagger(a_expr): get_expectation_a_dagger(),
            a_expr: get_expectation_a(),
        }
    )


def expectation_from_expr(expr: sp.Expr) -> sp.Expr:
    """Get the expectation of an expression which is written in terms of the generators."""
    return expectation_from_formula(formula_from_expr(expr))


expect_x = expectation_from_expr(x_expr)
expect_x_squared = expectation_from_expr((x_expr - expect_x) ** 2)
expect_x_squared = sp.simplify(expect_x_squared)

expect_p = expectation_from_expr(p_expr)
expect_p_squared = expectation_from_expr((p_expr - expect_p) ** 2)
expect_p_squared = sp.simplify(expect_p_squared)

uncertainty_squared = expect_p_squared * expect_x_squared
uncertainty_squared = sp.factor(sp.expand(uncertainty_squared))
