from __future__ import annotations

import sympy as sp


def expression_equals(expr1: sp.Expr, expr2: sp.Expr | int) -> bool:
    """Check if two expressions are equal, ignoring the order of terms."""
    return sp.simplify(expr1 - expr2, rational=True) == 0  # type: ignore[no-untyped-call]
