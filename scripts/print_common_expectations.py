from __future__ import annotations

import sympy as sp

from three_variable.coherent_states import (
    expect_p,
    expect_p_squared,
    expect_x,
    expect_x_squared,
    expectation_from_formula,
    uncertainty_squared,
)
from three_variable.equilibrium_squeeze import squeeze_ratio_from_zeta_expr
from three_variable.symbols import a_dagger_expr, a_expr, k_0, k_minus, k_plus

print("Expect X:")
sp.print_latex(sp.simplify(expect_x))
print()
print("Expect X^2:")
sp.print_latex(expect_x_squared)
print()
sp.print_latex(squeeze_ratio_from_zeta_expr(expect_x_squared))
print()


print("Expect P:")
sp.print_latex(sp.simplify(expect_p))
print()
print("Expect P^2: ")
sp.print_latex(expect_p_squared)
print()
sp.print_latex(squeeze_ratio_from_zeta_expr(expect_p_squared))
print()

print("Uncertainty:")
sp.print_latex(uncertainty_squared)
print()
sp.print_latex(sp.factor(squeeze_ratio_from_zeta_expr(uncertainty_squared)))
print()

print("Expect a:")
sp.print_latex(expectation_from_formula(a_expr))
print("Expect a^dagger:")
sp.print_latex(expectation_from_formula(a_dagger_expr))
print()


print("Expect K_0:")
sp.print_latex(expectation_from_formula(k_0))
print()

print("Expect K_+:")
sp.print_latex(expectation_from_formula(k_plus))
print()

print("Expect K_-:")
sp.print_latex(expectation_from_formula(k_minus))
print()
