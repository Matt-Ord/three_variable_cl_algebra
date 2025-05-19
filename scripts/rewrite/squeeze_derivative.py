from __future__ import annotations

import sympy as sp

from three_variable.new_paper.equilibrium_squeeze import (
    get_equilibrium_squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.new_paper.projected_sse import (
    get_environment_derivative,
    get_system_deriviative,
)
from three_variable.new_paper.symbols import dimensionless_from_full, hbar

expr_system = sp.simplify(dimensionless_from_full(get_system_deriviative("zeta")))
print("System derivative:")
sp.print_latex(expr_system)
expr_r_system = sp.factor(squeeze_ratio_from_zeta_expr(expr_system))
sp.print_latex(expr_r_system)

expr_environment = dimensionless_from_full(get_environment_derivative("zeta"))
print("Environment derivative:")
sp.print_latex(expr_environment)
expr_r_environment = sp.factor(squeeze_ratio_from_zeta_expr(expr_environment))
sp.print_latex(expr_r_environment)

expr_r_full = sp.factor_terms(
    (-sp.I / hbar) * expr_r_system - expr_r_environment, fraction=True
)
sp.print_latex(expr_r_full)


sp.print_latex(get_equilibrium_squeeze_ratio())
