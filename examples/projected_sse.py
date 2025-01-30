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
noise = sp.symbols("N")

state_creator = CreateBoson(0)
state_annihilator = Dagger(state_creator)


@cache
def create_vaccum_boson(i: float) -> sp.Expr:
    b = sp.exp(sp.I * phi) * CreateBoson(i) + alpha
    return mu * b + nu * Dagger(b)


@cache
def annihilate_vaccum_boson(i: float) -> sp.Expr:
    return Dagger(create_vaccum_boson(i))


lambda_, tau = sp.symbols(("lambda", "tau"), real=True, positive=True)


@cache
def get_lindblad_operator(i: float) -> sp.Expr:
    return sp.sqrt(lambda_ / (2 * tau**2)) * (
        (tau**2 - 1) * create_vaccum_boson(i)
        - (tau**2 + 1) * annihilate_vaccum_boson(i)
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


def get_hamiltonian_term(i: float) -> sp.Expr:
    creator = create_vaccum_boson(i)
    annihilator = annihilate_vaccum_boson(i)
    return sp.sqrt(lambda_) * (creator**2 - annihilator**2)


diffusion_term = get_diffusion_term(0)
drift_term = get_drift_term(0)
hamiltonian_shift_term = get_hamiltonian_term(0)

sse = hamiltonian_shift_term + drift_term + diffusion_term


# expectation = expectation.expand()
# muls = expectation.atoms(sp.Mul)

vaccum = FockStateBosonKet([0])
a = apply_operators(Dagger(vaccum) * sse * vaccum)


# def factor_mu_nu(expr: sp.Expr) -> sp.Expr:
#     expr = expr.expand()
#     # muls = e.atoms(Mul)
#     # subs_list = [(m, _apply_Mul(m)) for m in iter(muls)]
#     # return e.subs(subs_list)
#     # A = sp.Wild("A")
#     # expr = expr.replace()
#     sp.Wild("A")

#     # return expr.rewrite([mu, nu])

#     return sp.simplify(
#         expr.subs(
#             {
#                 mu: sp.symbols("mu"),
#                 nu: sp.symbols("nu"),
#             }
#         )
#     )


# drift = (
#     drift.collect(nu * sp.conjugate(nu))
#     .subs({nu * sp.conjugate(nu): mu**2 - 1})
#     .expand()
# )


for i in range(3):
    print()
    print("------------------------------------------")
    print("Expectations for i =", i)
    state = FockStateBosonKet([i])

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
