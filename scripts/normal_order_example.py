from __future__ import annotations

import sympy as sp

from three_variable.projected_sse import (
    get_diffusion_term,
    get_drift_term,
    get_harmonic_term,
    get_kinetic_term,
    get_lindblad_expectation,
)
from three_variable.symbols import formula_from_expr, p_expr, x_expr

kinetic = get_kinetic_term()
print(formula_from_expr(kinetic))

harmonic = get_harmonic_term()
print(formula_from_expr(harmonic))


uncertainty_x = x_expr**2
uncertainty_p = p_expr**2
print(formula_from_expr(uncertainty_x))
print(formula_from_expr(uncertainty_p))
print(formula_from_expr(uncertainty_p))

hamiltonian = sp.Add(get_kinetic_term(), get_harmonic_term())
print(formula_from_expr(hamiltonian))

drift = get_drift_term()
print(formula_from_expr(drift))

diffusion = get_diffusion_term()
print(formula_from_expr(diffusion))

print(get_lindblad_expectation())
