from __future__ import annotations

import sympy as sp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.operatorordering import normal_ordered_form

from three_variable.projected_sse import (
    get_diffusion_term,
    get_drift_term,
    get_hamiltonian_shift_term,
    get_harmonic_term,
    get_kinetic_term,
    get_lindblad_operator,
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
sp.print_latex(sp.simplify(formula_from_expr(p_expr**2), rational=True))
print()

print("Kinetic term:")
kinetic = get_kinetic_term()
print(formula_from_expr(kinetic))

print("Harmonic term:")
harmonic = get_harmonic_term()
print(formula_from_expr(harmonic))

print("Shift term:")
shift = get_hamiltonian_shift_term()
sp.print_latex(sp.simplify(formula_from_expr(shift), rational=True))
print()

print("Lindblad term:")
lindblad = get_lindblad_operator()
sp.print_latex(formula_from_expr(lindblad))
print()

print("Lindblad product term:")
sp.print_latex(sp.factor(formula_from_expr(Dagger(lindblad) * lindblad)))  # type: ignore unknown
print()

print("Drift term:")
drift = get_drift_term()
sp.print_latex(formula_from_expr(drift))
print()

print("Diffusion term:")
diffusion = get_diffusion_term()
sp.print_latex(formula_from_expr(diffusion))
print()


print(normal_ordered_form(a_expr * a_dagger_expr + a_expr * a_expr))
print(normal_ordered_form(a_expr * (a_dagger_expr + a_expr)))
