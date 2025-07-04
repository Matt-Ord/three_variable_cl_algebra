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

print("The Raw derivatives which describe the alpha parameter:")
expr_system = sp.simplify(get_system_derivative("alpha"))  # type: ignore subs
expr_system = expr_system.subs({sp.Symbol("V_1"): 0})  # type: ignore subs
expr_system = sp.factor(expr_system)  # type: ignore unknown
print("System derivative:")
sp.print_latex(expr_system)  # type: ignore subs
print()

expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
expr_r_system = sp.collect(  # type: ignore collect
    sp.factor(sp.expand(expr_r_system)),  # type: ignore subs
    [alpha, sp.conjugate(alpha)],  # type: ignore subs
)
print("System derivative (R):")
sp.print_latex(expr_r_system)  # type: ignore subs
print()

expr_environment = get_environment_derivative("alpha")
expr_environment = sp.simplify(sp.together(sp.expand(expr_environment)))  # type: ignore subs
print("Environment derivative:")
expr_environment = sp.collect(
    expr_environment, [sp.Symbol("alpha"), sp.conjugate(sp.Symbol("alpha"))]
)
sp.print_latex(expr_environment)

expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))  # type: ignore subs
expr_r_environment = sp.collect(  # type: ignore collect
    sp.factor(sp.expand(expr_r_environment)),  # type: ignore subs
    [alpha, sp.conjugate(alpha)],  # type: ignore subs
)
print("Environment derivative (R):")
sp.print_latex(expr_r_environment)  # type: ignore subs
print()
