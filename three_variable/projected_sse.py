from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

import sympy as sp
from slate_core.util import timed
from sympy.physics.quantum import (
    Dagger,
)
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    action_from_expr,
    expect_p,
    expect_x,
    expectation_from_expr,
    extract_action,
    xp_expression_from_alpha,
)
from three_variable.symbols import (
    KBT,
    a_dagger_expr,
    a_expr,
    alpha,
    dimensionless_from_full,
    eta_m,
    lambda_,
    m,
    m_from_eta_m,
    noise,
    omega,
    p_expr,
    x_expr,
)
from three_variable.util import file_cached


def _get_lindblad_operator_raw() -> sp.Expr:
    return (sp.sqrt(lambda_ * (4 * m * KBT) / (hbar**2)) * x_expr) + (
        1j * sp.sqrt(lambda_ / (4 * m * KBT)) * p_expr
    )


def _get_lindblad_operator_simple() -> sp.Expr:
    return sp.sqrt(lambda_ / (4 * eta_m)) * (
        (2 * eta_m - 1) * a_dagger_expr + (2 * eta_m + 1) * a_expr
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
        Dagger(lindblad_expectation) * lindblad_operator  # type: ignore sp
        - (Dagger(lindblad_operator) * lindblad_operator) / 2
        - (Dagger(lindblad_expectation) * lindblad_expectation) / 2
    )


def get_hamiltonian_shift_term() -> sp.Expr:
    shift = lambda_ * (x_expr * p_expr + p_expr * x_expr) / 2
    return sp.simplify(normal_ordered_form(shift))


def get_kinetic_term() -> sp.Expr:
    return (p_expr * p_expr) / (2 * m)


def get_x_polynomial(n: int) -> sp.Expr:
    return (x_expr - (alpha / sp.sqrt(2))) ** n


def get_linear_term() -> sp.Expr:
    return sp.Symbol("V_1") * x_expr


def get_harmonic_term() -> sp.Expr:
    return 0.5 * m * omega**2 * x_expr**2


def _get_system_derivative_path(ty: Literal["zeta", "alpha", "phi"]) -> Path:
    return Path(f".cache/system_derivative.{ty}")


@file_cached(_get_system_derivative_path)
@timed
def get_system_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    linear_term = get_linear_term()
    kinetic_term = get_kinetic_term()
    harmonic_term = get_harmonic_term()
    expr = (-1j / hbar) * (linear_term + kinetic_term + harmonic_term)

    return sp.simplify(
        dimensionless_from_full(sp.factor(extract_action(action_from_expr(expr), ty))),
        rational=True,
    )


def _get_environment_derivative(ty: Literal["zeta", "alpha", "phi"]) -> Path:
    return Path(f".cache/environment_derivative.{ty}")


@file_cached(_get_environment_derivative)
@timed
def get_environment_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    drift_term = get_drift_term()
    environment_derivative = sp.factor(extract_action(action_from_expr(drift_term), ty))

    shift_expr = (-1j / hbar) * get_hamiltonian_shift_term()
    shift_derivative = sp.factor(extract_action(action_from_expr(shift_expr), ty))

    return sp.simplify(
        dimensionless_from_full(shift_derivative + environment_derivative),
        rational=True,
    )


def get_deterministic_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    system_derivative = get_system_derivative(ty)
    environment_derivative = get_environment_derivative(ty)
    return system_derivative + environment_derivative


@cache
@timed
def get_stochastic_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    expr = get_diffusion_term()

    return sp.simplify(
        dimensionless_from_full(sp.factor(extract_action(action_from_expr(expr), ty))),
        rational=True,
    )


@cache
@timed
def get_full_derivative(ty: Literal["zeta", "alpha", "phi"]) -> sp.Expr:
    deterministic_derivative = get_deterministic_derivative(ty)
    stochastic_derivative = get_stochastic_derivative(ty)
    return deterministic_derivative + stochastic_derivative


def _get_classical_deterministic_derivative(ty: Literal["x", "p"]) -> Path:
    return Path(f".cache/classical_deterministic_derivative.{ty}")


@file_cached(_get_classical_deterministic_derivative)
@timed
def get_classical_deterministic_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    # Write <x> and <p> in terms of the real and imaginary parts of alpha.
    # Then replace re(alpha) and im(alpha) with their derivatives.
    # This gives us d/dt <x> etc.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative_xp = xp_expression_from_alpha(get_deterministic_derivative("alpha"))

    expect_var = expect_x if ty == "x" else expect_p
    expect_var = expect_var.subs({alpha: sp.re(alpha) + 1j * sp.im(alpha)})

    return sp.simplify(
        expect_var.subs(
            {sp.re(alpha): sp.re(derivative_xp), sp.im(alpha): sp.im(derivative_xp)}
        ),
        rational=True,
    )


def _get_classical_stochastic_derivative(ty: Literal["x", "p"]) -> Path:
    return Path(f".cache/classical_stochastic_derivative.{ty}")


@file_cached(_get_classical_stochastic_derivative)
@timed
def get_classical_stochastic_derivative(
    ty: Literal["x", "p"],
) -> sp.Expr:
    # Write <x> and <p> in terms of the real and imaginary parts of alpha.
    # Then replace re(alpha) and im(alpha) with their derivatives.
    # This gives us d/dt <x> etc.
    # Note this assumes d \zeta / dt = 0, which is true at equilibrium.
    derivative_xp = xp_expression_from_alpha(get_stochastic_derivative("alpha"))

    expect_var = expect_x if ty == "x" else expect_p
    expect_var = expect_var.subs({alpha: sp.re(alpha) + 1j * sp.im(alpha)})

    return sp.simplify(
        expect_var.subs(
            {sp.re(alpha): sp.re(derivative_xp), sp.im(alpha): sp.im(derivative_xp)}
        ),
        rational=True,
    )
