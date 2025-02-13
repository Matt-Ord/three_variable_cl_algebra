from __future__ import annotations

import sympy as sp

from three_variable.equilibrium_squeeze import (
    eta_lambda,
    eta_m,
    eta_omega,
    get_classical_alpha_derivative,
    get_equilibrium_squeeze_beta,
    get_equilibrium_squeeze_derivative,
    get_equilibrium_squeeze_derivative_gradient,
    get_equilibrium_squeeze_R,
)
from three_variable.symbols import noise

print("get_equilibrium_squeeze_R")
sp.print_latex(get_equilibrium_squeeze_R())
print()
print("get_equilibrium_squeeze_beta")
sp.print_latex(get_equilibrium_squeeze_beta())
print()

print("Free Particle Limit")
sp.print_latex(sp.limit(get_equilibrium_squeeze_R(), eta_omega, sp.oo))
print()
print("Low Friction Limit")
sp.print_latex(sp.series(get_equilibrium_squeeze_R(), eta_lambda, sp.oo, n=2))
print()
print("Low Friction Free particle")
sp.print_latex(
    sp.series(
        sp.limit(get_equilibrium_squeeze_R(), eta_omega, sp.oo), eta_lambda, sp.oo, n=1
    )
)
print()
print("High Friction Free particle")
sp.print_latex(
    sp.series(
        sp.limit(get_equilibrium_squeeze_R(), eta_omega, sp.oo), eta_lambda, 0, n=2
    )
)
print()
print("High Friction")
sp.print_latex(sp.series(get_equilibrium_squeeze_R(), eta_lambda, 0, n=2))
print()

assert get_equilibrium_squeeze_derivative() == 0

sp.print_latex(get_equilibrium_squeeze_derivative_gradient())


print("Classical Alpha derivative")
alpha_derivative = get_classical_alpha_derivative()
sp.print_latex(alpha_derivative)

alpha_derivative_parts = sp.collect(sp.expand(alpha_derivative), noise, evaluate=False)
high_mass_limit = sum(
    sp.sqrtdenest(
        sp.simplify(sp.together(sp.expand(k * next(v.lseries(eta_m, sp.oo)))))
    )
    for k, v in alpha_derivative_parts.items()
)
sp.print_latex(high_mass_limit)
