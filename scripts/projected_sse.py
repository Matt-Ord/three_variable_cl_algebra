from __future__ import annotations

import sympy as sp
from sympy.physics.secondquant import (
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

from three_variable.equilibrium_squeeze import get_squeeze_derivative_R
from three_variable.projected_sse import (
    get_diffusion_term,
    get_drift_term,
    get_hamiltonian_shift_term,
    get_harmonic_term,
    get_kinetic_term,
    get_linear_term,
    get_squeeze_derivative_beta,
    get_squeeze_derivative_environment_beta,
    get_squeeze_derivative_system_beta,
)
from three_variable.symbols import (
    alpha,
    eta_m,
    hbar,
)

diffusion_term = get_diffusion_term(0)
drift_term = get_drift_term(0)
hamiltonian_shift_term = get_hamiltonian_shift_term(0)
kinetic_term = get_kinetic_term(0)
harmonic_term = get_harmonic_term(0)
linear_term = get_linear_term(0)

sp.print_latex(get_squeeze_derivative_system_beta())
print()
sp.print_latex(get_squeeze_derivative_environment_beta())
print()
sp.print_latex(get_squeeze_derivative_beta())
print()
sp.print_latex(get_squeeze_derivative_R())
input()


print()
print("------------------------------------------")
print("Expectations for i =", 0)
state = FockStateBosonKet([0])
vaccum = FockStateBosonKet([0])

print("Linear")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * linear_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)
print("Harmonic")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * harmonic_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)

print()
print("------------------------------------------")
print("Expectations for i =", 1)
state = FockStateBosonKet([1])
vaccum = FockStateBosonKet([0])


print("Kinetic")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * kinetic_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)
print("Linear")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * linear_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)
print("Harmonic")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * harmonic_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)

print("Hamiltonian Shift + Drift")
sp.print_latex(
    sp.collect(
        sp.factor_terms(
            apply_operators(
                Dagger(state)
                * ((-sp.I / hbar) * hamiltonian_shift_term + drift_term)
                * vaccum
            )
        ),
        [alpha, sp.conjugate(alpha)],
    )
)


print("Diffusion")
sp.print_latex(
    sp.factor_terms(
        sp.collect(apply_operators(Dagger(state) * diffusion_term * vaccum), eta_m)
    )
)

print("------------------------------------------")

print("------------------------------------------")
print("Expectations for i =", 2)
state = FockStateBosonKet([2])
vaccum = FockStateBosonKet([0])

print("Kinetic")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * kinetic_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)
print("Linear")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * linear_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)
print("Harmonic")
sp.print_latex(
    sp.collect(
        sp.factor_terms(apply_operators(Dagger(state) * harmonic_term * vaccum)),
        [alpha, sp.conjugate(alpha)],
    )
)

print("Hamiltonian Shift + Drift")
poly = sp.Poly(
    sp.factor_terms(
        eta_m
        * apply_operators(
            Dagger(state)
            * (((-sp.I / hbar) * hamiltonian_shift_term) + drift_term)
            * vaccum
        )
    ),
    eta_m,
)
terms = [(monom, sp.factor(coeff)) for monom, coeff in poly.terms()]
poly = sp.Poly.from_dict(dict(terms), *poly.gens)


sp.print_latex(sp.factor_terms(poly.as_expr() / eta_m))


print("Diffusion")
sp.print_latex(
    sp.factor_terms(
        sp.collect(apply_operators(Dagger(state) * diffusion_term * vaccum), eta_m)
    )
)

print("------------------------------------------")
