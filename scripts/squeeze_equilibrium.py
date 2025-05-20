from __future__ import annotations

import itertools

import sympy as sp

from three_variable.coherent_states import expect_x_squared
from three_variable.equilibrium_squeeze import (
    get_equilibrium_squeeze_ratio,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.symbols import (
    eta_lambda,
    eta_omega,
)
from three_variable.util import print_latex

expect_x_squared_ratio = squeeze_ratio_from_zeta_expr(expect_x_squared)


print("-------------------------------------------------")
print("The equilibrium squeeze ratio:")
equilibrium_ratio = get_equilibrium_squeeze_ratio()
print_latex(equilibrium_ratio)
print()


print()
print("Low Friction Limit")
low_friction = equilibrium_ratio.lseries(eta_lambda, sp.oo)
low_friction_series = sum(sp.simplify(e) for e in itertools.islice(low_friction, 3))
print("Ratio: ", end="")
print_latex(low_friction_series)
print("Width: ", end="")
print_latex(expect_x_squared_ratio.subs({squeeze_ratio: low_friction_series}))
print()
print("High Friction Limit")
print("Ratio: ", end="")
print_latex(sp.series(equilibrium_ratio, eta_lambda, 0, n=2))
print()

print("------------------------------------------------")
print("Free Particle Limit")
print_latex(sp.limit(equilibrium_ratio, eta_omega, sp.oo))
print()
print("Low Friction Free particle")
print_latex(
    sp.series(sp.limit(equilibrium_ratio, eta_omega, sp.oo), eta_lambda, sp.oo, n=1)
)
print()
print("High Friction Free particle")
print_latex(
    sp.series(sp.limit(equilibrium_ratio, eta_omega, sp.oo), eta_lambda, 0, n=2)
)
