from __future__ import annotations

import sympy as sp

from three_variable.coherent_states import expect_x_squared
from three_variable.equilibrium_squeeze import (
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import (
    get_environment_derivative,
    get_system_derivative,
)

expect_x_squared_ratio = squeeze_ratio_from_zeta_expr(expect_x_squared)

print("The Raw derivatives which describe the squeeze parameter:")
expr_system = sp.simplify(get_system_derivative("zeta"))
print("System derivative:")
sp.print_latex(expr_system)
expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
print("System derivative (R):")
sp.print_latex(expr_r_system)
print()

expr_environment = get_environment_derivative("zeta")
print("Environment derivative:")
sp.print_latex(expr_environment)
expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))
print("Environment derivative (R):")
sp.print_latex(expr_r_environment)
print()

expr_r_full = sp.factor_terms(expr_r_system + expr_r_environment, fraction=True)
print("Full derivative (R):")
sp.print_latex(sp.nsimplify(expr_r_full, rational=True))
