from __future__ import annotations

from functools import cache

import sympy as sp
from sympy.physics.secondquant import (
    CreateBoson,
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

from three_variable.symbols import (
    KBT,
    alpha,
    beta,
    eta_m,
    hbar,
    lambda_,
    lambda_from_eta_lambda,
    m,
    m_from_eta_m,
    mu,
    noise,
    nu,
    omega,
    omega_from_eta_omega,
    phi,
)

from .decorators import timed


@cache
def create_vaccum_boson(i: float) -> sp.Expr:
    b = CreateBoson(i) + alpha
    return mu * b - nu * Dagger(b)


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


def _get_lindblad_operator_raw(i: float) -> sp.Expr:
    return (sp.sqrt(lambda_ * (4 * m * KBT) / (hbar**2)) * get_x_operator(i)) + (
        sp.I * sp.sqrt(lambda_ / (4 * m * KBT)) * get_p_operator(i)
    )


def _get_lindblad_operator_simple(i: float) -> sp.Expr:
    return sp.sqrt(lambda_ / (4 * eta_m)) * (
        (2 * eta_m + 1) * create_vaccum_boson(i)
        + (2 * eta_m - 1) * annihilate_vaccum_boson(i)
    )


@cache
def get_lindblad_operator(i: float) -> sp.Expr:
    # Sanity check - does the raw and simple form of the lindblad operator match?
    raw = _get_lindblad_operator_raw(i).subs({m: m_from_eta_m})
    simple = _get_lindblad_operator_simple(i)
    diff = sp.simplify(sp.expand(raw - simple))
    assert diff == 0
    return simple


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
        Dagger(lindblad_expectation) * lindblad_operator
        - (Dagger(lindblad_operator) * lindblad_operator) / 2
        - (Dagger(lindblad_expectation) * lindblad_expectation) / 2
    )


def _get_hamiltonian_shift_term_raw(i: float) -> sp.Expr:
    x = get_x_operator(i)
    p = get_p_operator(i)
    return lambda_ * (x * p + p * x) / 2


def _get_hamiltonian_shift_term_simple(i: float) -> sp.Expr:
    creator = create_vaccum_boson(i)
    annihilator = annihilate_vaccum_boson(i)
    return -sp.I * lambda_ * hbar * (creator**2 - annihilator**2) / 2


def get_hamiltonian_shift_term(i: float) -> sp.Expr:
    # TODO: Sanity check - does the raw and simple form of the hamiltonian shift term match?
    # For this we need to be able to use normal ordering
    raw = _get_hamiltonian_shift_term_raw(i)
    simple = _get_hamiltonian_shift_term_simple(i)
    assert sp.simplify(sp.expand(raw - simple)) == 0
    return simple


def get_kinetic_term(i: float) -> sp.Expr:
    p = get_p_operator(i)
    return (p * p) / (2 * m)


def get_x_polynomial(i: float, n: int) -> sp.Expr:
    x = sp.Symbol("x")
    x_0 = sp.Symbol("x_0")
    c = sp.Symbol("C")
    polynomial = sp.Integer(1)
    for _ in range(n):
        polynomial = (x - x_0) * polynomial - c * sp.Derivative(
            polynomial, x, evaluate=True
        )
    x_0_expr = (
        alpha * (mu - sp.conjugate(nu)) + sp.conjugate(alpha) * (mu - nu)
    ) / sp.sqrt(2)
    c_expr = (mu - nu) * (mu - sp.conjugate(nu)) / 2
    return polynomial.subs({x: get_x_operator(i), x_0: x_0_expr, c: c_expr})


def get_linear_term(i: float) -> sp.Expr:
    return get_x_polynomial(i, 1)


def get_harmonic_term(i: float) -> sp.Expr:
    return get_x_polynomial(i, 2)


@cache
@timed
def get_squeeze_derivative_system() -> sp.Expr:
    state = FockStateBosonKet([2])
    vaccum = FockStateBosonKet([0])

    linear_term = sp.Symbol("V_1") * get_linear_term(0)
    kinetic_term = get_kinetic_term(0)
    harmonic_term = (m * omega**2) * get_harmonic_term(0) / 2

    out = sp.factor_terms(
        apply_operators(
            Dagger(state) * (kinetic_term + linear_term + harmonic_term) * vaccum
        )
    )
    subbed = (-sp.I / hbar) * out.subs(
        {
            phi: 0,
            m: m_from_eta_m,
            lambda_: lambda_from_eta_lambda,
            omega: omega_from_eta_omega,
        }
    )
    prefactor = (mu**2 / 2) * (1 / sp.sqrt(2))
    out = sp.simplify(prefactor * subbed)
    neumer, denom = sp.together(out).as_numer_denom()
    collected = sp.collect(sp.expand(neumer), eta_m, evaluate=False)
    neumer = sum(k * sp.factor(v) for k, v in collected.items())
    return neumer / denom


@cache
def get_squeeze_derivative_system_beta() -> sp.Expr:
    derivative = get_squeeze_derivative_system()
    subbed = derivative.subs({nu: beta * mu})
    return sp.factor_terms(subbed)


@cache
@timed
def get_squeeze_derivative_environment() -> sp.Expr:
    state = FockStateBosonKet([2])
    vaccum = FockStateBosonKet([0])

    drift_term = get_drift_term(0)
    hamiltonian_shift_term = get_hamiltonian_shift_term(0)

    out = sp.factor_terms(
        apply_operators(
            Dagger(state)
            * (((-sp.I / hbar) * hamiltonian_shift_term) + drift_term)
            * vaccum
        )
    )
    subbed = out.subs(
        {
            phi: 0,
            m: m_from_eta_m,
            lambda_: lambda_from_eta_lambda,
            omega: omega_from_eta_omega,
        }
    )
    prefactor = (mu**2 / 2) * (1 / sp.sqrt(2))
    out = sp.simplify(prefactor * subbed)
    neumer, denom = sp.together(out).as_numer_denom()
    collected = sp.collect(sp.expand(neumer), eta_m, evaluate=False)
    return sp.factor_terms(sum(k * sp.factor(v) for k, v in collected.items())) / denom


@cache
def get_squeeze_derivative_environment_beta() -> sp.Expr:
    derivative = get_squeeze_derivative_environment()
    subbed = derivative.subs({nu: beta * mu})
    return sp.factor_terms(subbed)


@cache
def get_squeeze_derivative() -> sp.Expr:
    return sp.factor_terms(
        get_squeeze_derivative_system() + get_squeeze_derivative_environment()
    )


@cache
@timed
def get_squeeze_derivative_beta() -> sp.Expr:
    return sp.together(
        get_squeeze_derivative_system_beta() + get_squeeze_derivative_environment_beta()
    )


@cache
@timed
def get_alpha_derivative_system() -> sp.Expr:
    state = FockStateBosonKet([1])
    vaccum = FockStateBosonKet([0])

    linear_term = sp.Symbol("V_1") * get_linear_term(0)
    kinetic_term = get_kinetic_term(0)
    harmonic_term = (m * omega**2) * get_harmonic_term(0) / 2

    out = sp.factor_terms(
        apply_operators(
            Dagger(state) * (kinetic_term + linear_term + harmonic_term) * vaccum
        )
    )
    subbed = (-sp.I / hbar) * out.subs(
        {
            phi: 0,
            m: m_from_eta_m,
            lambda_: lambda_from_eta_lambda,
            omega: omega_from_eta_omega,
        }
    )
    collected = sp.collect(
        sp.expand(sp.simplify(subbed)), sp.Symbol("V_1"), evaluate=False
    )
    return sum(k * sp.factor(v) for k, v in collected.items())


@cache
def get_alpha_derivative_system_beta() -> sp.Expr:
    derivative = get_alpha_derivative_system()
    subbed = derivative.subs({nu: beta * mu})
    return sp.factor_terms(subbed)


@cache
@timed
def get_alpha_derivative_environment() -> sp.Expr:
    state = FockStateBosonKet([1])
    vaccum = FockStateBosonKet([0])

    drift_term = get_drift_term(0)
    hamiltonian_shift_term = get_hamiltonian_shift_term(0)

    out = sp.factor_terms(
        apply_operators(
            Dagger(state)
            * (((-sp.I / hbar) * hamiltonian_shift_term) + drift_term)
            * vaccum
        )
    )
    subbed = out.subs(
        {
            phi: 0,
            m: m_from_eta_m,
            lambda_: lambda_from_eta_lambda,
            omega: omega_from_eta_omega,
        }
    )
    out = sp.simplify(subbed)
    neumer, denom = sp.together(out).as_numer_denom()
    neumer = sp.factor(neumer)
    return neumer / denom


@cache
def get_alpha_derivative_environment_beta() -> sp.Expr:
    derivative = get_alpha_derivative_environment()
    subbed = derivative.subs({nu: beta * mu})
    return sp.factor_terms(subbed)


@cache
@timed
def get_alpha_derivative_stochastic() -> sp.Expr:
    state = FockStateBosonKet([1])
    vaccum = FockStateBosonKet([0])

    diffusion_term = get_diffusion_term(0)
    out = sp.factor_terms(apply_operators(Dagger(state) * (diffusion_term) * vaccum))
    subbed = out.subs(
        {
            phi: 0,
            m: m_from_eta_m,
            lambda_: lambda_from_eta_lambda,
            omega: omega_from_eta_omega,
        }
    )
    return sp.simplify(subbed)


@cache
def get_alpha_derivative_stochastic_beta() -> sp.Expr:
    derivative = get_alpha_derivative_stochastic()
    subbed = derivative.subs({nu: beta * mu})
    return sp.factor_terms(subbed)


def get_alpha_derivative() -> sp.Expr:
    return (
        get_alpha_derivative_system()
        + get_alpha_derivative_environment()
        + get_alpha_derivative_stochastic()
    )


def get_alpha_derivative_beta() -> sp.Expr:
    return (
        get_alpha_derivative_system_beta()
        + get_alpha_derivative_environment_beta()
        + get_alpha_derivative_stochastic_beta()
    )


def get_x_derivative_classical() -> sp.Expr:
    x_operator = get_x_operator(0)
    p_operator = get_p_operator(0)
    vaccum = FockStateBosonKet([0])
    classical_x = apply_operators(Dagger(vaccum) * x_operator * vaccum)
    classical_p = apply_operators(Dagger(vaccum) * p_operator * vaccum)

    solution = sp.solve(
        [sp.Eq(sp.Symbol("x"), classical_x), sp.Eq(sp.Symbol("p"), classical_p)],
        (alpha, sp.conjugate(alpha)),
    )
    solution = {
        k: sp.expand(sp.simplify(v.subs({mu**2: 1 + nu * sp.conjugate(nu)})))
        for k, v in solution.items()
    }

    derivative_alpha = get_alpha_derivative()
    sp.print_latex(classical_x)

    x_derivative = sp.expand(
        sp.expand(
            classical_x.subs(
                {
                    alpha: derivative_alpha,
                    sp.conjugate(alpha): sp.conjugate(derivative_alpha),
                }
            )
        ).subs({mu**2: 1 + nu * sp.conjugate(nu)})
    )
    sp.print_latex(solution)
    return sp.simplify(x_derivative.subs(solution))
    Q = sp.symbols("Q")
    return sp.expand(x_derivative.subs({alpha: sp.conjugate(alpha) + Q}))
    factor_1 = (
        alpha * mu
        + alpha * sp.conjugate(mu)
        - mu * sp.conjugate(alpha)
        - nu * sp.conjugate(alpha)
    )
    sp.simplify(factor_1.subs(solution))
    return sp.simplify(x_derivative.subs(solution))
