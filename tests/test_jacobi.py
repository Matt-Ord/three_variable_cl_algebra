from __future__ import annotations

from sympy.physics.quantum import Commutator
from sympy.physics.units import hbar

from tests.util import expression_equals
from three_variable.symbols import (
    a_dagger_expr,
    a_expr,
    formula_from_expr,
    k_0,
    k_0_expr,
    k_minus,
    k_minus_expr,
    k_plus,
    k_plus_expr,
    p_expr,
    x_expr,
)


def test_commutators_su() -> None:
    # The commutators of our K operators must match the su(1,1) algebra
    # [K_0, K_-] = -K_-
    # [K_0, K_+] = K_+
    # [K_+, K_-] = -2K_0

    commutator = Commutator(k_0_expr, k_minus_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), -k_minus)

    commutator = Commutator(k_0_expr, k_plus_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), k_plus)

    commutator = Commutator(k_plus_expr, k_minus_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), -2 * k_0)


def test_commutators_heisenburg() -> None:
    # The commutators of a, a_dagger must match the Heisenberg algebra
    # [a, a_dagger] = 1 <- this ...

    commutator = Commutator(a_expr, a_dagger_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), 1)


def test_commutators_x() -> None:
    # The commutators of x, p must match the Heisenberg algebra
    # [x, p] = i * hbar <- this ...

    commutator = Commutator(x_expr, p_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), 1j * hbar)


def test_commutators_jacobi() -> None:
    # The commutators between the two algebra must match the Jacobi algebra
    # [a, K_+] = a dagger
    # [a dagger, K_-] = -a
    # [K_+, a dagger] = [K_-, a] = 0
    # [K_0, a dagger] = 0.5 * a dagger, [K_0, a] = -0.5 * a

    commutator = Commutator(a_expr, k_plus_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), a_dagger_expr)

    commutator = Commutator(a_dagger_expr, k_minus_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), -a_expr)

    commutator = Commutator(k_plus_expr, a_dagger_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), 0)

    commutator = Commutator(k_minus_expr, a_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), 0)

    commutator = Commutator(k_0_expr, a_dagger_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), 0.5 * a_dagger_expr)

    commutator = Commutator(k_0_expr, a_expr).doit()  # type: ignore[no-untyped-call]
    assert expression_equals(formula_from_expr(commutator), -0.5 * a_expr)
