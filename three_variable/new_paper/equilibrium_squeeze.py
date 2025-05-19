# Dimensionless Parameters for the squeezing operator analysis
from __future__ import annotations

from functools import cache

import sympy as sp

from .projected_sse import get_environment_derivative, get_system_deriviative
from .symbols import dimensionless_from_full, hbar, zeta

squeeze_ratio = sp.Symbol(r"R")
ratio_expr = (1 - zeta) / (1 + zeta)


def squeeze_ratio_from_zeta_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing parameter to the squeezing ratio."""
    zeta_expr = sp.solve(ratio_expr - squeeze_ratio, zeta)[0]
    return sp.simplify(expr.subs(zeta, zeta_expr))


def zeta_from_squeeze_ratio_expr(expr: sp.Expr) -> sp.Expr:
    """Convert the squeezing ratio to the squeezing parameter."""
    return sp.simplify(expr.subs(squeeze_ratio, ratio_expr))


@cache
def get_equilibrium_squeeze_ratio(*, positive: bool = False) -> sp.Expr:
    expr_system = sp.simplify(dimensionless_from_full(get_system_deriviative("zeta")))
    expr_environment = dimensionless_from_full(get_environment_derivative("zeta"))

    expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
    expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))

    factored = sp.factor_terms(
        (-sp.I / hbar) * expr_r_system - expr_r_environment, fraction=True
    )
    numer, _denom = sp.together(factored).as_numer_denom()
    return sp.solve(numer, squeeze_ratio)[0 if positive else 1]
