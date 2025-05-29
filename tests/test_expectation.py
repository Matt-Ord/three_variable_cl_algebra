from __future__ import annotations

import pytest
import sympy as sp
from sympy.physics.units import hbar

from tests.util import expression_equals
from three_variable.coherent_states import (
    action_from_expr,
    expectation_from_action,
    expectation_from_expr,
    get_expectation_a,
    get_expectation_a_dagger,
    get_expectation_k_0,
    get_expectation_k_minus,
    get_expectation_k_plus,
)
from three_variable.symbols import (
    a_dagger_expr,
    a_expr,
    alpha,
    k_0_expr,
    k_minus_expr,
    k_plus_expr,
    p_expr,
    x_expr,
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
        var, re_var + 1j * im_var
    )

    re_deriv = sp.Derivative(subbed, re_var).doit()  # type: ignore[no-untyped-call]
    im_deriv = sp.Derivative(subbed, im_var).doit()  # type: ignore[no-untyped-call]

    deriv = 0.5 * (re_deriv - 1j * im_deriv)
    return sp.simplify(
        deriv.subs(
            {
                re_var: 0.5 * (var + sp.conjugate(var)),
                im_var: -1j * 0.5 * (var - sp.conjugate(var)),
            }
        )
    )


def expectation_from_expr_test(action: sp.Expr) -> sp.Expr:
    return expectation_from_action(action_from_expr(action))


@pytest.mark.parametrize(
    ("expr", "expectation"),
    [
        (a_expr, get_expectation_a()),
        (a_dagger_expr, get_expectation_a_dagger()),
        (k_plus_expr, get_expectation_k_plus()),
        (k_0_expr, get_expectation_k_0()),
        (k_minus_expr, get_expectation_k_minus()),
        (x_expr, (get_expectation_a() + get_expectation_a_dagger()) / sp.sqrt(2)),
        (
            p_expr,
            -1j
            * hbar
            * (get_expectation_a() - get_expectation_a_dagger())
            / sp.sqrt(2),
        ),
    ],
)
def test_expectation_from_action(expr: sp.Expr, expectation: sp.Expr) -> None:
    from_action = expectation_from_expr_test(expr)
    assert expression_equals(from_action, expectation)
    from_expr = expectation_from_expr(expr)
    assert expression_equals(from_expr, expectation)


def test_expectation_from_kernel() -> None:
    """Test the expectation value of a from the kernel."""
    derivative = complex_wirtinger_derivative(sp.log(kernel_expr), alpha)
    derivative_alpha_expr = get_expectation_a()
    assert expression_equals(derivative, derivative_alpha_expr)

    derivative = complex_wirtinger_derivative(sp.log(kernel_expr), zeta)
    derivative_zeta_expr = get_expectation_k_minus()
    assert expression_equals(derivative, derivative_zeta_expr)


def test_expectation_a_dagger() -> None:
    """Test the expectation value of a dagger."""
    expectation_a = get_expectation_a()
    expectation_a_dagger = get_expectation_a_dagger()
    assert expression_equals(sp.conjugate(expectation_a), expectation_a_dagger)

    derivative_formula = alpha + zeta * expectation_a
    assert expression_equals(derivative_formula, expectation_a_dagger)


def test_expectation_k_plus() -> None:
    """Test the expectation value of k plus."""
    expectation_k_minus = get_expectation_k_minus()
    expectation_k_plus = get_expectation_k_plus()
    assert expression_equals(sp.conjugate(expectation_k_minus), expectation_k_plus)

    expectation_a = get_expectation_a()
    derivative_formula = (
        0.5 * alpha**2
        + 0.5 * zeta
        + alpha * zeta * expectation_a
        + zeta**2 * expectation_k_minus
    )
    assert expression_equals(derivative_formula, expectation_k_plus)


def test_get_expectation_k_0() -> None:
    """Test the expectation value of k zero."""
    expectation_k_0 = get_expectation_k_0()
    expectation_k_0 = sp.simplify(sp.factor(expectation_k_0))
    # Should be hermitian!
    assert expression_equals(sp.conjugate(expectation_k_0), expectation_k_0)
