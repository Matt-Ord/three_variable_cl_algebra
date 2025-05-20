# What factors of x should we use to expand the potential energy term?
# We want x powers that only contribute to nth order squeezing terms.
from __future__ import annotations

import sympy as sp

from three_variable.coherent_states import (
    action_from_expr,
    expectation_from_expr,
)
from three_variable.projected_sse import (
    get_x_polynomial,
)
from three_variable.symbols import alpha, formula_from_expr, p_expr, x_expr

print("x operator")
x_formula = formula_from_expr(x_expr)
sp.print_latex(x_formula)
x_action = action_from_expr(x_expr - (alpha / sp.sqrt(2)))
sp.print_latex(x_action)
x_expectation = expectation_from_expr(x_expr)
sp.print_latex(x_expectation)

print("x^2 operator")
x_expectation = formula_from_expr((x_expr - (alpha / sp.sqrt(2))) ** 2)
sp.print_latex(x_expectation)
x_action = action_from_expr((x_expr - (alpha / sp.sqrt(2))) ** 2)
sp.print_latex(sp.simplify(x_action))


p_expectation = expectation_from_expr(p_expr)
print("p operator")
sp.print_latex(p_expectation)
print()

x_polynomial = [get_x_polynomial(i) for i in range(3)]


for i, s in enumerate(x_polynomial):
    # We basically just want to avoid the x^2 term appearing in the SDE for the
    print(f"Polynomial {i}")
    sp.print_latex(sp.simplify(action_from_expr(s)))
