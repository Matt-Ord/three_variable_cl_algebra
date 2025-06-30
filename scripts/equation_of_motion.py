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
high_temperature
R_expr = sum(sp.simplify(e) for e in itertools.islice(low_friction, 1))
R = sp.Symbol("R", complex=True)

# sp.print_latex(R)
# Define zeta = (1 - R) / (1 + R)

# Compute modulus r = |zeta|
# r = sp.Abs(zeta)

# # Extract real and imaginary parts of zeta
# zeta_re = sp.re(zeta)
# zeta_im = sp.im(zeta)


# # Compute sin(theta/2) and cos(theta/2)
# sqrt_term = sp.sqrt(zeta_re**2 + zeta_im**2)
# sin_theta_2 = sp.sqrt((sqrt_term - zeta_re) / (2 * sqrt_term))
# cos_theta_2 = sp.sqrt((sqrt_term + zeta_re) / (2 * sqrt_term))

# # Define the diagonal squeezing matrix
# D = sp.Matrix([[sp.exp(-r), 0], [0, sp.exp(r)]])

# # Define the rotation matrix
# P = sp.Matrix([[cos_theta_2, sin_theta_2], [-sin_theta_2, cos_theta_2]])

# # Combine the transformation
# S = D * P
# S = sp.simplify(S.subs({zeta: (1 - R) / (1 + R)}))

# # S_series = S.applyfunc(
# #     lambda expr: sum(
# #         sp.simplify(e) for e in itertools.islice(expr.lseries(eta_lambda, sp.oo), 2)
# #     )
# # )

# # # Simplify and display
# # sp.print_latex(S_series)
# sp.print_latex(S)
# input()

F_expr = sp.simplify(get_system_derivative("alpha")).subs({sp.Symbol("V_1"): 0})
F_expr = sp.factor(F_expr) / alpha
F = sp.Symbol("F", complex=True)
G = sp.Symbol("G", complex=True)

alpha_conj = sp.conjugate(alpha)

# Define x and p in terms of alpha and alpha*
x = sp.Symbol("x", complex=True)
p = sp.Symbol("p", complex=True)

# Define time derivatives
dalpha_dt = F * alpha + G * alpha_conj
dalpha_conj_dt = sp.conjugate(F) * alpha_conj + sp.conjugate(G) * alpha

# Differentiate x and p manually using chain rule
dx_dt = dalpha_dt / sp.sqrt(2) + dalpha_conj_dt / sp.sqrt(2)
dp_dt = sp.I * dalpha_dt / sp.sqrt(2) - sp.I * dalpha_conj_dt / sp.sqrt(2)

# substitution
alpha_expr = (x - sp.I * p) / sp.sqrt(2)
alpha_conj_expr = (x + sp.I * p) / sp.sqrt(2)

dx_dt = dx_dt.subs(
    {alpha: alpha_expr, alpha_conj: alpha_conj_expr, zeta: (1 - R_expr) / (1 + R_expr)}
)
dp_dt = dp_dt.subs(
    {alpha: alpha_expr, alpha_conj: alpha_conj_expr, zeta: (1 - R_expr) / (1 + R_expr)}
)


# Collect terms in xp and pp
dx_dt_collected = sp.collect(sp.simplify(dx_dt), [x, p])
dp_dt_collected = sp.collect(sp.simplify(dp_dt), [x, p])

# Display
print("The time derivatives of x:")
sp.print_latex(dx_dt_collected)
print("The time derivatives of p:")
sp.print_latex(dp_dt_collected)

# Group into system
equations = [dx_dt_collected, dp_dt_collected]
variables = [x, p]

# Extract coefficient matrix M
M, b = sp.linear_eq_to_matrix(equations, variables)

# # Optional: verify everything
print("Coefficient matrix M:")
sp.print_latex(M)
