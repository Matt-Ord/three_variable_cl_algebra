from __future__ import annotations

import sympy as sp
from sympy.physics.units import hbar

from three_variable.coherent_states import expect_x_squared
from three_variable.equilibrium_squeeze import (
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import (
    get_environment_derivative,
    get_system_derivative,
)
from three_variable.util import factor_terms, print_latex, simplify

expect_x_squared_ratio = squeeze_ratio_from_zeta_expr(expect_x_squared)

print("The Raw derivaties which describe the squeeze parameter:")
expr_system = simplify(get_system_derivative("zeta"))
print("System derivative:")
print_latex(expr_system)
expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
print_latex(expr_r_system)

expr_environment = get_environment_derivative("zeta")
print("Environment derivative:")
print_latex(expr_environment)
expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))
print_latex(expr_r_environment)

expr_r_full = factor_terms(
    (-sp.I / hbar) * expr_r_system - expr_r_environment, fraction=True
)
print_latex(expr_r_full)
