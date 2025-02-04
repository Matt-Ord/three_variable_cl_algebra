from __future__ import annotations

from functools import cache

import sympy as sp
from sympy.physics.secondquant import (
    CreateBoson,
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

r, theta, phi = sp.symbols(("r", r"\theta", "phi"), real=True)
alpha = sp.symbols(r"\alpha", complex=True)
mu_expr = sp.cosh(r)
nu_expr = sp.exp(sp.I * theta) * sp.sinh(r)
mu = sp.symbols(r"\mu", real=True)
nu = sp.symbols(r"\nu", real=False)
noise = sp.symbols(r"\xi(t)", complex=True)
hbar, omega, m = sp.symbols(("hbar", "omega", "m"), real=True, positive=True)

state_creator = CreateBoson(0)
state_annihilator = Dagger(state_creator)


@cache
def create_vaccum_boson(i: float) -> sp.Expr:
    b = sp.exp(sp.I * phi) * CreateBoson(i) + alpha
    return mu * b + nu * Dagger(b)


@cache
def annihilate_vaccum_boson(i: float) -> sp.Expr:
    return Dagger(create_vaccum_boson(i))


def get_x_operator(i: float) -> sp.Expr:
    return (create_vaccum_boson(i) + annihilate_vaccum_boson(i)) / sp.sqrt(2)


def get_p_operator(i: float) -> sp.Expr:
    return (
        -sp.I
        * hbar
        * (create_vaccum_boson(i) - annihilate_vaccum_boson(i))
        / sp.sqrt(2)
    )


lambda_, tau = sp.symbols(("lambda", "tau"), real=True, positive=True)


@cache
def get_lindblad_operator(i: float) -> sp.Expr:
    return sp.sqrt(lambda_ / (2 * tau**2)) * (
        (tau**2 - 1) * create_vaccum_boson(i)
        + (tau**2 + 1) * annihilate_vaccum_boson(i)
    )


@cache
def get_lindblad_expectation(i: float) -> sp.Expr:
    lindblad_operator = get_lindblad_operator(i)
    vaccum = FockStateBosonKet([0])
    return apply_operators(Dagger(vaccum) * lindblad_operator * vaccum)


def get_diffusion_term(i: float) -> sp.Expr:
    lindblad_operator = get_lindblad_operator(i)
    lindblad_expectation = get_lindblad_expectation(i)
    return noise * (lindblad_operator - lindblad_expectation)


def get_drift_term(i: float) -> sp.Expr:
    lindblad_operator = get_lindblad_operator(i)
    lindblad_expectation = get_lindblad_expectation(i)
    return (
        sp.conjugate(lindblad_expectation) * lindblad_operator
        - (Dagger(lindblad_operator) * lindblad_operator) / 2
        - (sp.conjugate(lindblad_expectation) * lindblad_expectation) / 2
    )


def get_hamiltonian_shift_term(i: float) -> sp.Expr:
    creator = create_vaccum_boson(i)
    annihilator = annihilate_vaccum_boson(i)
    return sp.I * lambda_ * hbar * (creator**2 - annihilator**2) / 2


def get_kinetic_term(i: float) -> sp.Expr:
    p = get_p_operator(i)
    return (p * p) / (2 * m)


def get_linear_term(i: float) -> sp.Expr:
    x_shift = (
        alpha * (mu + sp.conjugate(nu)) + sp.conjugate(alpha) * (nu + mu)
    ) / sp.sqrt(2)
    return get_x_operator(i) - x_shift


def get_harmonic_term(i: float) -> sp.Expr:
    x = get_linear_term(i)
    return x**2 - (mu + nu) * (mu + sp.conjugate(nu)) / 2


diffusion_term = get_diffusion_term(0)
drift_term = get_drift_term(0)
hamiltonian_shift_term = get_hamiltonian_shift_term(0)
kinetic_term = get_kinetic_term(0)
harmonic_term = get_harmonic_term(0)
linear_term = get_linear_term(0)
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

print("Hamiltonian Shift")
sp.print_latex(
    sp.simplify(
        sp.factor_terms(
            apply_operators(Dagger(state) * hamiltonian_shift_term * vaccum)
        )
    )
)

print("Drift")
sp.print_latex(
    sp.factor_terms(
        sp.collect(apply_operators(Dagger(state) * drift_term * vaccum), tau)
    )
)

print("Diffusion")
sp.print_latex(
    sp.factor_terms(
        sp.collect(apply_operators(Dagger(state) * diffusion_term * vaccum), tau)
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

print("Hamiltonian Shift")
sp.print_latex(
    sp.simplify(
        sp.factor_terms(
            apply_operators(Dagger(state) * hamiltonian_shift_term * vaccum)
        )
    )
)

print("Drift")
sp.print_latex(
    sp.factor_terms(
        sp.collect(apply_operators(Dagger(state) * drift_term * vaccum), tau)
    )
)

print("Diffusion")
sp.print_latex(
    sp.factor_terms(
        sp.collect(apply_operators(Dagger(state) * diffusion_term * vaccum), tau)
    )
)

print("------------------------------------------")
