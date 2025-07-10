from __future__ import annotations

import sympy as sp

from three_variable.coherent_states import p, x, xp_expression_from_alpha
from three_variable.equilibrium_squeeze import get_equilibrium_squeeze_ratio
from three_variable.symbols import alpha, eta_lambda, zeta

equilibrium_ratio = get_equilibrium_squeeze_ratio()
classical_ratio = sp.limit(equilibrium_ratio, eta_lambda, sp.oo)
classical_zeta = (1 - classical_ratio) / (1 + classical_ratio)


F = sp.Symbol("F", complex=True)
G = sp.Symbol("G", complex=True)
dalpha_dt = F * alpha + G * sp.conjugate(alpha)
dalpha_conj_dt = sp.conjugate(F) * sp.conjugate(alpha) + sp.conjugate(G) * alpha

# Differentiate x and p manually using chain rule
dx_dt = dalpha_dt / sp.sqrt(2) + dalpha_conj_dt / sp.sqrt(2)
dp_dt = 1j * dalpha_dt / sp.sqrt(2) - 1j * dalpha_conj_dt / sp.sqrt(2)


dx_dt = xp_expression_from_alpha(dx_dt).subs({zeta: classical_zeta})
dp_dt = xp_expression_from_alpha(dp_dt).subs({zeta: classical_zeta})


dx_dt_collected = sp.collect(sp.simplify(dx_dt), [x, p])  # type: ignore collect
dp_dt_collected = sp.collect(sp.simplify(dp_dt), [x, p])  # type: ignore collect

print("The time derivatives of x:")
sp.print_latex(dx_dt_collected)
print("The time derivatives of p:")
sp.print_latex(dp_dt_collected)

print("Coefficient matrix M:")
M, b = sp.linear_eq_to_matrix([dx_dt_collected, dp_dt_collected], [x, p])  # type: ignore unknown
sp.print_latex(M)  # type: ignore unknown
