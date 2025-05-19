from __future__ import annotations

import sympy as sp

from three_variable.equilibrium_squeeze import get_squeeze_derivative_R
from three_variable.projected_sse import (
    get_alpha_derivative_environment_beta,
    get_alpha_derivative_stochastic_beta,
    get_alpha_derivative_system_beta,
    get_squeeze_derivative_beta,
    get_squeeze_derivative_environment_beta,
    get_squeeze_derivative_system_beta,
    get_x_derivative_classical,
)

print("Alpha derivative")
sp.print_latex(get_alpha_derivative_system_beta())
print()
sp.print_latex(get_alpha_derivative_environment_beta())
print()
sp.print_latex(get_alpha_derivative_stochastic_beta())
print()

print("Full Alpha derivative")
sp.print_latex(get_x_derivative_classical())

print("Squeeze derivative")
sp.print_latex(get_squeeze_derivative_system_beta())
print()
sp.print_latex(get_squeeze_derivative_environment_beta())
print()

print("Full Squeeze derivative")
sp.print_latex(get_squeeze_derivative_beta())
print()
sp.print_latex(get_squeeze_derivative_R())
