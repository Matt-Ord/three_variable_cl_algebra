from __future__ import annotations

import sympy as sp
from sympy.physics.secondquant import (
    CreateBoson,
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

# Define symbolic variables
alpha, beta = sp.symbols(r"\alpha \beta", complex=True)

creation_operator = CreateBoson(0)

state_0 = FockStateBosonKet([0])
state_1 = FockStateBosonKet([1])
state_2 = FockStateBosonKet([2])

state_phi = sp.I * state_0
state_alpha = (creation_operator + (sp.conjugate(alpha) / 2)) * state_0
state_beta = (
    creation_operator * creation_operator - (sp.conjugate(beta) / 2)
) * state_0

matrix = [
    [apply_operators(Dagger(bra) * ket) for ket in (state_phi, state_alpha, state_beta)]
    for bra in (state_0, state_1, state_2)
]
M_inv = sp.Matrix(matrix)
M = M_inv.inv()

sp.pprint(M_inv)
sp.pprint(M)

sp.print_latex(M_inv)
