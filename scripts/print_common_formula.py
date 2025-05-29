from __future__ import annotations

import sympy as sp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from three_variable.projected_sse import (
    get_diffusion_term,
    get_drift_term,
    get_harmonic_term,
    get_kinetic_term,
)
from three_variable.symbols import (
    a_dagger_expr,
    a_expr,
    formula_from_expr,
    p_expr,
    x_expr,
)

print("X:")
sp.print_latex(formula_from_expr(x_expr))
print()
print("X^2:")
sp.print_latex(formula_from_expr(x_expr**2))
print()

print("P:")
sp.print_latex(formula_from_expr(p_expr))
print()
print("P^2:")
sp.print_latex(formula_from_expr(p_expr**2))
print()

print("Kinetic term:")
kinetic = get_kinetic_term()
print(formula_from_expr(kinetic))

print("Harmonic term:")
harmonic = get_harmonic_term()
print(formula_from_expr(harmonic))

print("Drift term:")
drift = get_drift_term()
sp.print_latex(formula_from_expr(drift))

print("Diffusion term:")
diffusion = get_diffusion_term()
sp.print_latex(formula_from_expr(diffusion))


print(normal_ordered_form(a_expr * a_dagger_expr + a_expr * a_expr))
print(normal_ordered_form(a_expr * (a_dagger_expr + a_expr)))
