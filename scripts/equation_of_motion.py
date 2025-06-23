from __future__ import annotations

import itertools

import sympy as sp

from three_variable.equilibrium_squeeze import get_equilibrium_squeeze_ratio
from three_variable.projected_sse import (
    get_system_derivative,
)
from three_variable.symbols import (
    alpha,
    eta_lambda,
    zeta,
)

equilibrium_ratio = get_equilibrium_squeeze_ratio()
low_friction = equilibrium_ratio.lseries(eta_lambda, sp.oo)  # type: ignore sp
R = sum(sp.simplify(e) for e in itertools.islice(low_friction, 2))

sp.print_latex(R)
# input()

F = sp.simplify(get_system_derivative("alpha")).subs({sp.Symbol("V_1"): 0})
F = sp.factor(F) / alpha

alpha_conj = sp.conjugate(alpha)

# Define x' and p' in terms of alpha and alpha*
x = sp.Symbol("x", complex=True)
p = sp.Symbol("p", complex=True)

# Define time derivatives
dalpha_dt = F * alpha
dalpha_conj_dt = sp.conjugate(F) * alpha_conj

# Differentiate x' and p' manually using chain rule
dxp_dt = dalpha_dt / sp.sqrt(2) + dalpha_conj_dt / sp.sqrt(2)
dpp_dt = dalpha_dt / (sp.I * sp.sqrt(2)) - dalpha_conj_dt / (sp.I * sp.sqrt(2))

# Optional substitution
alpha_expr = (x + sp.I * p) / sp.sqrt(2)
alpha_conj_expr = (x - sp.I * p) / sp.sqrt(2)

dxp_dt = dxp_dt.subs(
    {alpha: alpha_expr, alpha_conj: alpha_conj_expr, zeta: (1 - R) / (1 + R)}
)
dpp_dt = dpp_dt.subs(
    {alpha: alpha_expr, alpha_conj: alpha_conj_expr, zeta: (1 - R) / (1 + R)}
)

# # Simplify final results
# dxp_dt_final = sp.sp.simplify(dxp_dt)
# dpp_dt_final = sp.simplify(dxp_dt)

# print("The time derivatives of x' and p':")
# print("dx'/dt:")
# sp.print_latex(dxp_dt_final)
# print("dp'/dt:")
# sp.print_latex(dpp_dt_final)

# Collect terms in xp and pp
dxp_dt_collected = sp.collect(sp.simplify(dxp_dt), [x, p])
dpp_dt_collected = sp.collect(sp.simplify(dpp_dt), [x, p])

# Display
sp.print_latex(dxp_dt_collected)
sp.print_latex(dpp_dt_collected)
