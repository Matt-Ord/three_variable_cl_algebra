from __future__ import annotations

import sympy as sp
from transformation import (
    get_s_x,
)

from three_variable.coherent_states import (
    p,
    x,
)

# # Define symbols
# lambda_, eta_m = sp.symbols("lambda_ eta_m", real=True, positive=True)
# a = Operator("a")
# adag = Dagger(a)

# # Define prefactor and L
# prefactor = sp.sqrt(lambda_ / (4 * eta_m))
# L = prefactor * ((2 * eta_m - 1) * adag + (2 * eta_m + 1) * a)
# Ldag = Dagger(L)

# # Define expectation values
# a_exp = sp.Symbol("⟨a⟩", complex=True)
# adag_exp = sp.conjugate(a_exp)

# # Define <L> and <L†> in terms of <a> and <a†>
# L_exp = prefactor * ((2 * eta_m - 1) * adag_exp + (2 * eta_m + 1) * a_exp)
# Ldag_exp = sp.conjugate(L_exp)

# # 1. Define expression
# expr = Ldag_exp * L - (1 / 2) * (Ldag * L + Ldag_exp * L_exp)

# # 2. Expand Ldag*L and normal-order manually
# # First expand L†L
# LdagL = sp.expand(Ldag * L)

# # Apply bosonic commutator: a * adag = adag * a + 1
# LdagL = LdagL.replace(
#     lambda expr: isinstance(expr, sp.Mul) and a in expr.args and adag in expr.args,
#     lambda expr: expr.subs(a * adag, adag * a + 1),
# )

# # Final expression
# expr_final = Ldag_exp * L - (1 / 2) * (LdagL + Ldag_exp * L_exp)

# Define symbols (real positive where needed)
m, kB, T, lam, hbar = sp.symbols("m kB T lambda hbar", real=True, positive=True)


# Define L and L^\dagger
A = sp.sqrt(4 * m * kB * T * lam / hbar**2)
B = sp.sqrt(lam / (4 * m * kB * T))

L = A * x + sp.I * B * p
Ldag = A * x - sp.I * B * p  # Hermitian conjugate

# Define expectation symbols (commutative scalars)
x_exp = sp.Symbol("⟨x⟩", real=True)
p_exp = sp.Symbol("⟨p⟩", real=True)

L_exp = A * x_exp + sp.I * B * p_exp
Ldag_exp = A * x_exp - sp.I * B * p_exp

# Build the expression
expr = Ldag_exp * L - (1 / 2) * (Ldag * L + Ldag_exp * L_exp)

# Expand and simplify
expr = sp.expand(expr)

# transformation
S_x = get_s_x()
x_transformed = S_x[0, 0] * x + S_x[0, 1] * p
p_transformed = S_x[1, 0] * x + S_x[1, 1] * p
# Substitute transformed variables
expr = expr.subs({x: x_transformed, p: p_transformed})

# Optional: simplify and display
expr = sp.simplify(sp.expand(expr))
sp.print_latex(expr)
