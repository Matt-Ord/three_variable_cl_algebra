from __future__ import annotations

from typing import Literal

import sympy as sp
from sympy.physics.quantum import Dagger

from three_variable.new_paper.symbols import (
    a_expr,
    alpha,
    formula_from_expr,
    k_0,
    k_minus,
    k_plus,
    zeta,
)

kernel_prefactor_expr = (1 - sp.Abs(sp.conjugate(zeta)) ** 2) ** (-1 / 2)
kernel_exp_expr = sp.exp(
    (
        2 * sp.Abs(alpha) ** 2
        + alpha**2 * sp.conjugate(zeta)
        + sp.conjugate(alpha) ** 2 * zeta
    )
    / (2 * (1 - sp.Abs(sp.conjugate(zeta)) ** 2))
)
kernel_expr = (
    (kernel_prefactor_expr * kernel_exp_expr)
    .subs(sp.Abs(alpha) ** 2, alpha * sp.conjugate(alpha))
    .subs(sp.Abs(zeta) ** 2, zeta * sp.conjugate(zeta))
)


def complex_wirtinger_derivative(expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
    """Compute the derivative of a complex expression with respect to a complex variable."""
    re_var = sp.Symbol(f"re_{var.name}", real=True)
    im_var = sp.Symbol(f"im_{var.name}", real=True)
    subbed = expr.subs(sp.Abs(var) ** 2, var * sp.conjugate(var)).subs(
        var, re_var + sp.I * im_var
    )

    re_deriv = sp.Derivative(subbed, re_var).doit()
    im_deriv = sp.Derivative(subbed, im_var).doit()

    deriv = 0.5 * (re_deriv - sp.I * im_deriv)
    return sp.simplify(
        deriv.subs(
            {
                re_var: 0.5 * (var + sp.conjugate(var)),
                im_var: -sp.I * 0.5 * (var - sp.conjugate(var)),
            }
        )
    )


d_alpha = sp.Symbol("d_alpha")
d_zeta = sp.Symbol("d_zeta")

DIFFERENTIAL_ACTION_A = d_alpha
DIFFERENTIAL_ACTION_A_DAGGER = alpha + zeta * d_alpha
DIFFERENTIAL_ACTION_K_MINUS = d_zeta
DIFFERENTIAL_ACTION_K_PLUS = (
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
    expr = sp.collect(sp.expand(expr), [d_zeta, d_alpha])
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
    return EXPECTATION_D_ALPHA


def get_expectation_a_dagger() -> sp.Expr:
    """Compute < a_dagger >.

    a_dagger has the differential action of d / dalpha, so
    < a_dagger > = d / d alpha (ln(N)).
    """
    return (alpha + sp.conjugate(alpha) * zeta) / (1 - zeta * sp.conjugate(zeta))


def get_expectation_k_minus() -> sp.Expr:
    """Compute < k_- >.

    K_- has the differential action of d / dzeta, so
    < k_- > = d / dzeta (ln(N)).
    """
    return EXPECTATION_D_ZETA


def get_expectation_k_plus() -> sp.Expr:
    """Compute < k_+ >.

    K_+ has the differential action of d / dzeta, so
    < k_+ > = d / dzeta (ln(N)).
    """
    return (zeta / (2 * (1 - zeta * sp.conjugate(zeta)))) + (
        get_expectation_a_dagger() ** 2 / 2
    )


def get_expectation_k_0() -> sp.Expr:
    """Compute < k_+ >.

    K_+ has the differential action of d / dzeta, so
    < k_+ > = d / dzeta (ln(N)).
    """
    return 0.25 + (0.5 * alpha * get_expectation_a()) + zeta * get_expectation_k_minus()


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
