from __future__ import annotations

from sympy.physics.quantum import Commutator, qapply
from sympy.physics.quantum.boson import BosonFockBra, BosonFockKet

from three_variable.new_paper.projected_sse import (
    get_diffusion_term,
    get_drift_term,
    get_harmonic_term,
    get_kinetic_term,
    get_lindblad_expectation,
    p_expr,
    projection_operator_expr,
    x_expr,
)
from three_variable.new_paper.symbols import a_expr, formula_from_expr

kinetic = get_kinetic_term()
print(formula_from_expr(kinetic))

harmonic = get_harmonic_term()
print(formula_from_expr(harmonic))


comm = x_expr
for _i in range(5):
    comm = Commutator(projection_operator_expr, comm).doit()
    print(formula_from_expr(qapply(BosonFockBra(0) * comm * BosonFockKet(0))))
input()
print(
    formula_from_expr(
        qapply(
            BosonFockBra(0)
            * Commutator(projection_operator_expr, a_expr).doit()
            * BosonFockKet(0)
        )
    )
)
print(
    formula_from_expr(
        qapply(
            BosonFockBra(0)
            * Commutator(
                projection_operator_expr,
                Commutator(projection_operator_expr, a_expr).doit(),
            ).doit()
            * BosonFockKet(0)
        )
    )
)
input()
uncertainty_x = x_expr**2
uncertainty_p = p_expr**2
print(formula_from_expr(uncertainty_x))
print(formula_from_expr(uncertainty_p))
print(formula_from_expr(uncertainty_p))

lindblad = get_kinetic_term() + get_harmonic_term()
print(formula_from_expr(lindblad))

drift = get_drift_term()
print(formula_from_expr(drift))

diffusion = get_diffusion_term()
print(formula_from_expr(diffusion))

print(get_lindblad_expectation())
