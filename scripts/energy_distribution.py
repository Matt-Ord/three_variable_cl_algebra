from __future__ import annotations

import sympy as sp

from three_variable.projected_sse import (
    get_deterministic_derivative,
    get_stochastic_derivative,
)
from three_variable.symbols import (
    noise,
)

alpha_noise_im = get_stochastic_derivative("alpha").subs(noise, 1j)
alpha_noise_re = get_stochastic_derivative("alpha").subs(noise, 1)
alpha_deter = get_deterministic_derivative("alpha")
sp.print_latex(alpha_deter)

input()
alpha_derivative_deterministic = explicit_from_dimensionless(
    get_deterministic_derivative("alpha"), params
)

alpha_derivative_diffusion_re = explicit_from_dimensionless(
    get_stochastic_derivative("alpha").subs(noise, 1), params
)

alpha_derivative_diffusion_im = explicit_from_dimensionless(
    get_stochastic_derivative("alpha").subs(noise, 1j), params
)

zeta_derivative_deterministic = explicit_from_dimensionless(
    get_full_derivative("zeta"), params
).subs(noise, 0)

# Generate a matrix equation for the drift and diffusion terms
# And turn them into numpy ufuncs for the stochastic differential equation solver
drift_expr = sp.Matrix([alpha_derivative_deterministic, zeta_derivative_deterministic])
diff_expr = sp.Matrix(
    [[alpha_derivative_diffusion_re, alpha_derivative_diffusion_im], [0, 0]]
)

t = sp.Symbol("t", real=True)
drift_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), drift_expr, modules="numpy")  # type: ignore[no-redef]
diff_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), diff_expr, modules="numpy")  # type: ignore[no-redef]
