from __future__ import annotations

from functools import cache
from typing import Literal

import sympy as sp
from slate.util import timed
from sympy.physics.quantum import (
    Dagger,
)

from three_variable.symbols import (
    KBT,
    a_dagger_expr,
    a_expr,
    alpha,
    dimensionless_from_full,
    eta_m,
    hbar,
    lambda_,
    m,
    m_from_eta_m,
    noise,
    omega,
    p_expr,
    x_expr,
)

from .coherent_states import action_from_expr, expectation_from_expr, extract_action


def _get_lindblad_operator_raw() -> sp.Expr:
    return (sp.sqrt(lambda_ * (4 * m * KBT) / (hbar**2)) * x_expr) + (
        sp.I * sp.sqrt(lambda_ / (4 * m * KBT)) * p_expr
    )


def _get_lindblad_operator_simple() -> sp.Expr:
    return sp.sqrt(lambda_ / (4 * eta_m)) * (
        (2 * eta_m + 1) * a_dagger_expr + (2 * eta_m - 1) * a_expr
    )


@cache
def get_lindblad_operator() -> sp.Expr:
    # Sanity check - does the raw and simple form of the lindblad operator match?
    raw = _get_lindblad_operator_raw().subs({m: m_from_eta_m})
    simple = _get_lindblad_operator_simple()
    diff = sp.simplify(sp.expand(raw - simple))
    assert diff == 0
    return simple


@cache
def get_lindblad_expectation() -> sp.Expr:
    lindblad_operator = get_lindblad_operator()
    return expectation_from_expr(lindblad_operator)


def get_diffusion_term() -> sp.Expr:
    lindblad_operator = get_lindblad_operator()
    lindblad_expectation = get_lindblad_expectation()
    return noise * (lindblad_operator - lindblad_expectation)


def get_drift_term() -> sp.Expr:
    lindblad_operator = get_lindblad_operator()
    lindblad_expectation = get_lindblad_expectation()
    return (
        sp.conjugate(lindblad_expectation) * lindblad_operator
        - (Dagger(lindblad_operator) * lindblad_operator) / 2
        - (Dagger(lindblad_expectation) * lindblad_expectation) / 2
    )


def _get_hamiltonian_shift_term_raw() -> sp.Expr:
    return lambda_ * (x_expr * p_expr + p_expr * x_expr) / 2


def _get_hamiltonian_shift_term_simple() -> sp.Expr:
    return -sp.I * lambda_ * hbar * (a_dagger_expr**2 - a_expr**2) / 2


def get_hamiltonian_shift_term() -> sp.Expr:
    # Sanity check - does the raw and simple form of the hamiltonian shift term match?
    # For this we need to be able to use normal ordering
    raw = _get_hamiltonian_shift_term_raw()
    simple = _get_hamiltonian_shift_term_simple()
    assert sp.simplify(sp.expand(raw - simple)) == 0
    return simple


def get_kinetic_term() -> sp.Expr:
    return (p_expr * p_expr) / (2 * m)


def get_x_polynomial(n: int) -> sp.Expr:
    return (x_expr - (alpha / sp.sqrt(2))) ** n


def get_linear_term() -> sp.Expr:
    return sp.Symbol("V_1") * x_expr


def get_harmonic_term() -> sp.Expr:
    return 0.5 * m * omega**2 * x_expr**2


@cache
@timed
def get_system_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    linear_term = get_linear_term()
    kinetic_term = get_kinetic_term()
    harmonic_term = get_harmonic_term()
    expr = (-sp.I / hbar) * (linear_term + kinetic_term + harmonic_term)

    return sp.simplify(
        dimensionless_from_full(sp.factor(extract_action(action_from_expr(expr), ty))),
        rational=True,
    )


@cache
@timed
def get_environment_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    drift_term = get_drift_term()
    diffusion_term = get_diffusion_term()
    expr = drift_term + diffusion_term
    environment_derivative = sp.factor(extract_action(action_from_expr(expr), ty))

    shift_expr = (-sp.I / hbar) * get_hamiltonian_shift_term()
    shift_derivative = sp.factor(extract_action(action_from_expr(shift_expr), ty))

    return sp.simplify(
        dimensionless_from_full(shift_derivative + environment_derivative),
        rational=True,
    )
