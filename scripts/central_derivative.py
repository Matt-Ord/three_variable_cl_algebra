from __future__ import annotations

import sympy as sp

from three_variable.projected_sse import (
    get_environment_derivative,
    get_system_derivative,
)

print("The Raw derivaties which describe the alpha parameter:")
expr_system = sp.simplify(get_system_derivative("alpha")).subs({sp.Symbol("V_1"): 0})
expr_system = sp.factor(expr_system)
print("System derivative:")
sp.print_latex(expr_system)
print()

expr_environment = get_environment_derivative("alpha")
expr_environment = sp.simplify(sp.together(sp.expand(expr_environment)))
print("Environment derivative:")
# sp.print_latex(expr_environment)
expr_environment = sp.collect(
    expr_environment, [sp.Symbol("alpha"), sp.conjugate(sp.Symbol("alpha"))]
)
sp.print_latex(expr_environment)

# x = sp.Symbol("x", real=True)
# p = sp.Symbol("p", real=True)

# expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))
# expr_r_environment = sp.collect(
#     sp.factor(sp.expand(expr_r_environment)), [alpha, sp.conjugate(alpha)]
# )
# print("Environment derivative (R):")
# sp.print_latex(expr_r_environment)
# print()
