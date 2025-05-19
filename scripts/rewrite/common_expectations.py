from __future__ import annotations

import sympy as sp

from three_variable.new_paper.coherent_states import expectation_from_expr
from three_variable.new_paper.equilibrium_squeeze import squeeze_ratio_from_zeta_expr
from three_variable.new_paper.projected_sse import p_expr, x_expr

expect_x = expectation_from_expr(x_expr)
print("Expect X:")
sp.print_latex(sp.simplify(expect_x))
print()

expect_p = expectation_from_expr(p_expr)
print("Expect P:")
sp.print_latex(sp.simplify(expect_p))
print()

expect_x_squared = expectation_from_expr((x_expr - expect_x) ** 2)
expect_x_squared = sp.simplify(expect_x_squared)
print("Expect X^2:")
sp.print_latex(expect_x_squared)
print()
sp.print_latex(squeeze_ratio_from_zeta_expr(expect_x_squared))
print()

expect_p_squared = expectation_from_expr((p_expr - expect_p) ** 2)
expect_p_squared = sp.simplify(expect_p_squared)
print("Expect P^2:")
sp.print_latex(expect_p_squared)
print()
sp.print_latex(squeeze_ratio_from_zeta_expr(expect_p_squared))
print()

uncertainty_squared = expect_p_squared * expect_x_squared
uncertainty_squared = sp.factor(sp.expand(uncertainty_squared))
print("Uncertainty:")
sp.print_latex(uncertainty_squared)
print()
sp.print_latex(sp.factor(squeeze_ratio_from_zeta_expr(uncertainty_squared)))
print()
