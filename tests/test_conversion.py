from __future__ import annotations

import pytest
import sympy as sp
from sympy.physics.units import hbar

from tests.util import expression_equals
from three_variable.coherent_states import (
    alpha_expression_from_xp,
    expect_p,
    expect_x,
    p,
    x,
    xp_expression_from_alpha,
)
from three_variable.equilibrium_squeeze import (
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
    zeta_from_squeeze_ratio_expr,
)
from three_variable.symbols import (
    a_dagger_expr,
    a_expr,
    alpha,
    dimensionless_from_full,
    expr_from_formula,
    formula_from_expr,
    full_from_dimensionless,
    k_0,
    k_0_expr,
    k_minus,
    k_minus_expr,
    k_plus,
    k_plus_expr,
    lambda_,
    m,
    omega,
    p_expr,
    x_expr,
    zeta,
)


@pytest.mark.parametrize(
    ("symbol"),
    [(m), (omega), (lambda_)],
)
def test_conversion_dimensionless(symbol: sp.Symbol) -> None:
    # The conversion between dimensionless and dimensionful variables must be consistent

    assert (
        sp.simplify(symbol - full_from_dimensionless(dimensionless_from_full(symbol)))
        == 0
    )


def test_conversion_ratio() -> None:
    assert (
        sp.simplify(
            squeeze_ratio
            - squeeze_ratio_from_zeta_expr(zeta_from_squeeze_ratio_expr(squeeze_ratio))
        )
        == 0
    )
    assert (
        sp.simplify(
            zeta - zeta_from_squeeze_ratio_expr(squeeze_ratio_from_zeta_expr(zeta))
        )
        == 0
    )


@pytest.mark.parametrize(
    ("expr"),
    [a_dagger_expr, a_expr, k_0_expr, k_minus_expr, k_plus_expr],
)
def test_conversion_formula_simple(expr: sp.Expr) -> None:
    assert expression_equals(
        expr_from_formula(formula_from_expr(expr)),
        expr,
    )


def test_conversion_formula() -> None:
    assert expression_equals(
        formula_from_expr(2 * x_expr**2),
        formula_from_expr(
            a_expr**2 + a_dagger_expr**2 + 1 + 2 * a_dagger_expr * a_expr
        ),
    )
    assert expression_equals(formula_from_expr(x_expr**2), k_plus + k_minus + 2 * k_0)

    assert expression_equals(
        formula_from_expr(2 * p_expr**2 / hbar**2),
        formula_from_expr(
            -(a_expr**2) - a_dagger_expr**2 + 1 + 2 * a_dagger_expr * a_expr
        ),
    )
    assert expression_equals(
        formula_from_expr(p_expr**2 / hbar**2), -k_plus - k_minus + 2 * k_0
    )


def test_xp_from_alpha() -> None:
    """Test the conversion between x, p and alpha."""
    assert expression_equals(x, xp_expression_from_alpha(expect_x))
    assert expression_equals(p, xp_expression_from_alpha(expect_p))

    expr = alpha + zeta * sp.conjugate(alpha)
    assert expression_equals(
        expr, alpha_expression_from_xp(xp_expression_from_alpha(expr))
    )
