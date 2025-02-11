# What factors of x should we use to expand the potential energy term?
# We want x powers that only contribute to nth order squeezing terms.
from __future__ import annotations

import sympy as sp
from sympy.physics.secondquant import (
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

from three_variable.projected_sse import get_x_polynomial

x_operators = [get_x_polynomial(0, i) for i in range(5)]

for i in range(len(x_operators)):
    state = FockStateBosonKet([i])
    vaccum = FockStateBosonKet([0])
    print("--------------------------------------")
    print(f"n={i}")
    for s in x_operators:
        # Should only be non-zero if N_s = i
        sp.print_latex(sp.simplify(apply_operators(Dagger(state) * s * vaccum)))
