from __future__ import annotations

import itertools

import sympy as sp

from three_variable.equilibrium_squeeze import (
    R,
    eta_lambda,
    eta_m,
    eta_omega,
    get_classical_alpha_derivative,
    get_equilibrium_squeeze_beta,
    get_equilibrium_squeeze_derivative,
    get_equilibrium_squeeze_derivative_gradient,
    get_equilibrium_squeeze_R,
    get_uncertainty_x_R,
)
from three_variable.symbols import noise

print("get_equilibrium_squeeze_R")
sp.print_latex(get_equilibrium_squeeze_R())
print()
print("get_equilibrium_squeeze_beta")
sp.print_latex(get_equilibrium_squeeze_beta())
print()

print("--------------------------------------")
print("Free Particle Limit")
print("R")
sp.print_latex(sp.limit(get_equilibrium_squeeze_R(), eta_omega, sp.oo))
print("Delta X")
delta_x = sp.simplify(
    get_uncertainty_x_R()
    .subs({sp.re(R): (R + sp.conjugate(R)) / 2})
    .subs({R: (1 - sp.Symbol("A")) / (2 * eta_m)})
)
sp.print_latex(delta_x)
print("low friction free particle")
a_expr = sp.sqrt(4 * sp.I * eta_lambda + 2)
a_series = sp.together(
    sum(sp.simplify(e) for e in itertools.islice(a_expr.lseries(eta_lambda, sp.oo), 3))
)
sp.print_latex(a_series)
sp.print_latex(
    sum(
        sp.simplify(e)
        for e in itertools.islice(
            sp.simplify(delta_x.subs({sp.Symbol("A"): a_series})).lseries(
                eta_lambda, sp.oo
            ),
            3,
        )
    )
)
print("--------------------------------------")
print()
print("Low Friction Limit")
low_friction = get_equilibrium_squeeze_R().lseries(eta_lambda, sp.oo)
sp.print_latex(sum(sp.simplify(e) for e in itertools.islice(low_friction, 3)))
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
print("High Omega")
sp.print_latex(sp.series(get_equilibrium_squeeze_R(), eta_omega, 0, n=4))
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
