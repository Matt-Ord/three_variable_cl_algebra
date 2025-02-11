from __future__ import annotations

import sympy as sp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

t = sp.Symbol("t")
alpha = sp.Symbol("alpha")
a = BosonOp("a", annihilation=False)
A = sp.conjugate(alpha) * a - Dagger(sp.conjugate(alpha) * a)

a.hilbert_space

expA = sp.exp(A)  # Exponential of non-commutative operator A
derivative = sp.diff(expA, alpha)  # Compute derivative

# e^{\alpha {b^\dagger_{0}} + \overline{\alpha} b_{0}} \left(\frac{d}{d \alpha} \overline{\alpha} b_{0} + {b^\dagger_{0}}\right)
sp.print_latex(derivative)


def get_derivative_duhamel(a: sp.Expr, x: sp.Symbol) -> sp.Expr:
    s = sp.Symbol("s")
    return sp.exp(a) * sp.integrate(
        sp.exp(-s * a) * sp.Derivative(a, x) * sp.exp(s * a), (s, 0, 1)
    )


sp.print_latex(get_derivative_duhamel(A, alpha))
