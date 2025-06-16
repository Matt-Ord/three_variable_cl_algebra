from __future__ import annotations

import sympy as sp

from three_variable.equilibrium_squeeze import (
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import (
    get_environment_derivative,
    get_system_derivative,
)
from three_variable.symbols import alpha

print("The Raw derivaties which describe the alpha parameter:")
expr_system = sp.simplify(get_system_derivative("alpha")).subs({sp.Symbol("V_1"): 0})
expr_system = sp.factor(expr_system)
print("System derivative:")
sp.print_latex(expr_system)
print()

expr_environment = get_environment_derivative("alpha")
expr_environment = sp.simplify(sp.together(sp.expand(expr_environment)))
print("Environment derivative:")
sp.print_latex(expr_environment)

x = sp.Symbol("x", real=True)
p = sp.Symbol("p", real=True)

expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))
# expr_r_environment = expr_r_environment.subs(
#     {
#         alpha: (x - p),
#         sp.conjugate(alpha): (x + p),
#     }
# )
expr_r_environment = sp.collect(
    sp.factor(sp.expand(expr_r_environment)), [alpha, sp.conjugate(alpha)]
)
print("Environment derivative (R):")
sp.print_latex(expr_r_environment)
print()
