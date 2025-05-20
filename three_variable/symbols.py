from __future__ import annotations

from typing import cast

import sympy as sp
from sympy.physics.quantum import Dagger  # type: ignore sp
from sympy.physics.quantum.boson import BosonOp  # type: ignore sp
from sympy.physics.quantum.operatorordering import (  # type: ignore sp
    normal_ordered_form,  # type: ignore sp
)
from sympy.physics.units import hbar

# Physical constants
noise = sp.Symbol(r"\xi(t)")
lambda_ = sp.Symbol("lambda", real=True, positive=True)
omega = sp.Symbol("omega", real=True, positive=True)
m = sp.Symbol("m", real=True, positive=True)
KBT = sp.Symbol("KBT", real=True, positive=True)

# Phi and alpha for the coherent state
phi = sp.Symbol(r"\phi", real=True)
alpha = sp.Symbol(r"\alpha")
# Definitions for the squeezing operator
zeta = sp.Symbol(r"\zeta", real=False)


# Dimensionless Parameters for the squeezing operator analysis
eta_m = sp.Symbol("eta_m", real=True, positive=True)
eta_lambda = sp.Symbol("eta_lambda", real=True, positive=True)
eta_omega = sp.Symbol("eta_omega", real=True, positive=True)

m_from_eta_m = sp.Mul(sp.Pow(hbar, 2), eta_m, sp.Pow(sp.Mul(2, KBT), -1))
lambda_from_eta_lambda = sp.Mul(KBT, sp.Pow(sp.Mul(hbar, eta_lambda), -1))
omega_from_eta_omega = sp.Mul(KBT, sp.Pow(sp.Mul(hbar, eta_omega), -1))


def dimensionless_from_full(expr: sp.Expr) -> sp.Expr:
    """Convert the full expression to dimensionless form."""
    return expr.subs(  # type: ignore sp
        {
            m: m_from_eta_m,
            lambda_: lambda_from_eta_lambda,
            omega: omega_from_eta_omega,
        }
    )  # type: ignore sp


def full_from_dimensionless(expr: sp.Expr) -> sp.Expr:
    """Convert the dimensionless expression to full form."""
    eta_m_from_m = sp.solve(m_from_eta_m - m, eta_m)[0]  # type: ignore sp
    eta_lambda_from_lambda = sp.solve(lambda_from_eta_lambda - lambda_, eta_lambda)[0]  # type: ignore sp
    eta_omega_from_omega = sp.solve(omega_from_eta_omega - omega, eta_omega)[0]  # type: ignore sp
    return expr.subs(  # type: ignore sp
        {
            eta_m: eta_m_from_m,
            eta_lambda: eta_lambda_from_lambda,
            eta_omega: eta_omega_from_omega,
        }
    )


# Defines the expressions for the explicit squeezing and displacement operators
a_expr = BosonOp("a")
a_dagger_expr = cast("sp.Expr", Dagger(a_expr))
k_plus_expr = sp.Mul(0.5, sp.Pow(Dagger(a_expr), 2))  # type: ignore sp
k_0_expr = sp.Mul(0.5, sp.Add(0.5, sp.Mul(Dagger(a_expr), a_expr)))  # type: ignore sp
k_minus_expr = sp.Mul(0.5, sp.Pow(a_expr, 2))

x_expr = sp.Mul(sp.Add(a_expr, a_dagger_expr), 1 / sp.sqrt(2))  # type: ignore sp
p_expr = sp.Mul(-sp.I * hbar * (a_dagger_expr - a_expr), 1 / sp.sqrt(2))  # type: ignore sp

# The operators in symbol form
k_plus = sp.Symbol("K_+", commutative=False)
k_0 = sp.Symbol("K_0", commutative=False)
k_minus = sp.Symbol("K_-", commutative=False)


def formula_from_expr(expr: sp.Expr) -> sp.Expr:
    """Get the expression in terms of the five basic operators."""
    expr = normal_ordered_form(sp.expand(expr))  # type: ignore sp
    return sp.simplify(  # type: ignore sp
        expr.subs(  # type: ignore sp
            {
                Dagger(a_expr) * a_expr: (2 * k_0 - 0.5),  # type: ignore sp
                Dagger(a_expr) ** 2: 2 * k_plus,  # type: ignore sp
                a_expr**2: 2 * k_minus,  # type: ignore sp
            }
        )
    )


def expr_from_formula(expr: sp.Expr) -> sp.Expr:
    """Get the full expression, in terms of only the bosonic creation operator a."""
    return expr.subs(  # type: ignore sp
        {
            k_0: k_0_expr,
            k_plus: k_plus_expr,
            k_minus: k_minus_expr,
        }
    )
