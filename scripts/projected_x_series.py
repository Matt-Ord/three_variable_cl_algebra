# What factors of x should we use to expand the potential energy term?
# We want x powers that only contribute to nth order squeezing terms.
from __future__ import annotations

import sympy as sp
from sympy.physics.secondquant import (
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

from three_variable.projected_sse import (
    get_p_operator,
    get_x_operator,
    get_x_polynomial,
)

vaccum = FockStateBosonKet([0])

print("x operator")
x_operator = get_x_operator(0)
x_expression = apply_operators(Dagger(vaccum) * x_operator * vaccum)
sp.print_latex(x_expression)

p_operator = get_p_operator(0)
p_expression = apply_operators(Dagger(vaccum) * p_operator * vaccum)
print("p operator")
sp.print_latex(p_expression)


x_polynomial = [get_x_polynomial(0, i) for i in range(5)]

for i in range(len(x_polynomial)):
    state = FockStateBosonKet([i])
    vaccum = FockStateBosonKet([0])
    print("--------------------------------------")
    print(f"n={i}")
    for s in x_polynomial:
        # This is a test of our method for calculating the operator series
        # Should only be non-zero if N_s = i
        sp.print_latex(sp.simplify(apply_operators(Dagger(state) * s * vaccum)))
