from __future__ import annotations

import pytest
import sympy as sp

from three_variable.new_paper.coherent_states import (
    action_from_expr,
    complex_wirtinger_derivative,
    expectation_from_action,
    expectation_from_expr,
    get_expectation_a,
    get_expectation_a_dagger,
    get_expectation_k_0,
    get_expectation_k_minus,
    get_expectation_k_plus,
    kernel_expr,
)
from three_variable.new_paper.symbols import (
    a_dagger_expr,
    a_expr,
    alpha,
    k_0_expr,
    k_minus_expr,
    k_plus_expr,
    zeta,
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
    ],
)
def test_expectation_from_action(expr: sp.Expr, expectation: sp.Expr) -> None:
    from_action = expectation_from_expr_test(expr)
    assert sp.simplify(sp.factor(from_action - expectation)) == 0
    from_expr = expectation_from_expr(expr)
    assert sp.simplify(sp.factor(from_expr - expectation)) == 0


def test_expectation_from_kernel() -> None:
    """Test the expectation value of a from the kernel."""
    derivative = complex_wirtinger_derivative(sp.log(kernel_expr), alpha)
    derivative_alpha_expr = get_expectation_a()
    assert sp.simplify(sp.factor(derivative - derivative_alpha_expr)) == 0

    derivative = complex_wirtinger_derivative(sp.log(kernel_expr), zeta)
    derivative_zeta_expr = get_expectation_k_minus()
    assert (
        sp.simplify(sp.factor_terms(derivative - derivative_zeta_expr), rational=True)
        == 0
    )


def test_expectation_a_dagger() -> None:
    """Test the expectation value of a dagger."""
    expectation_a = get_expectation_a()
    expectation_a_dagger = get_expectation_a_dagger()
    assert sp.simplify(sp.conjugate(expectation_a) - expectation_a_dagger) == 0

    derivative_formula = alpha + zeta * expectation_a
    assert sp.simplify(derivative_formula - expectation_a_dagger) == 0


def test_expectation_k_plus() -> None:
    """Test the expectation value of k plus."""
    expectation_k_minus = get_expectation_k_minus()
    expectation_k_plus = get_expectation_k_plus()
    assert (
        sp.simplify(sp.factor(sp.conjugate(expectation_k_minus) - expectation_k_plus))
        == 0
    )

    expectation_a = get_expectation_a()
    derivative_formula = (
        0.5 * alpha**2
        + 0.5 * zeta
        + alpha * zeta * expectation_a
        + zeta**2 * expectation_k_minus
    )
    assert sp.simplify(derivative_formula - expectation_k_plus) == 0


def test_get_expectation_k_0() -> None:
    """Test the expectation value of k zero."""
    expectation_k_0 = get_expectation_k_0()
    expectation_k_0 = sp.simplify(sp.factor(expectation_k_0))
    sp.print_latex(sp.simplify(sp.factor(expectation_k_0)))
    # Should be hermitian!
    assert sp.simplify(sp.factor(sp.conjugate(expectation_k_0) - expectation_k_0)) == 0
