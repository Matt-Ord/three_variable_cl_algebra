# The pointer states are eigenstates of the matrix A-mu b, were
# A is the deterministic and B is the non-deterministic part of the SSE
# Can we predict how these change with mu? this should give us an
# idea of how <B> etc change with mu too.
from __future__ import annotations

from typing import override

import sympy as sp
from sympy.physics.quantum import Dagger, Ket, Operator


class AntiHermitianOperator(Operator):
    @override
    def _eval_adjoint(self) -> sp.Expr:
        return -self


L = Operator("L")

simple_a_part = AntiHermitianOperator("A_s")

mu = sp.Symbol("mu")
d_mu = sp.Symbol(r"\partial \mu")
perturbed_mu = sp.Add(mu, d_mu)


initial_state = Ket(r"n^\mu")
state_perturbation = Ket(r"{n'}^\mu")
other_state = Ket(r"m^\mu")
perturbed_state = sp.Add(initial_state, sp.Mul(d_mu, state_perturbation))
orthogonal_state = Ket(r"l^\mu")

initial_lambda = sp.Symbol(r"\lambda^\mu_n")
other_lambda = sp.Symbol(r"\lambda^\mu_m")


def _apply_drift_part(state: Ket) -> sp.Expr:
    return L * state * sp.Mul(Dagger(state), Dagger(L), state) - 0.5 * (  # type: ignore unknown
        (
            state  # type: ignore unknown
            * sp.Mul(Dagger(state), Dagger(L), state)  # type: ignore unknown
            * sp.Mul(Dagger(state), L, state)  # type: ignore unknown
        )
        + Dagger(L) * L * state  # type: ignore unknown
    )


def apply_a(state: Ket) -> sp.Expr:
    return sp.Mul(simple_a_part, state) + _apply_drift_part(state)  # type: ignore unknown


def apply_b(state: sp.Expr) -> sp.Expr:
    return sp.Mul(L, state) - state * sp.Mul(Dagger(state), L, state)  # type: ignore unknown


def apply_m(state: sp.Expr, mu: sp.Expr) -> sp.Expr:
    return apply_a(state) - mu * apply_b(state)  # type: ignore unknown


def extract_mu_factor(expr: sp.Expr, n: int) -> sp.Expr:
    expr = sp.expand(expr)  # type: ignore unknown
    out: sp.Expr = sp.Number(0)
    for i in range(n + 1):
        factor = d_mu**i * Dagger(d_mu) ** (n - i)  # type: ignore unknown
        try:
            out += sp.Mul(  # type: ignore unknown
                sp.collect(expr, [factor], exact=True, evaluate=False)[factor].subs(  # type: ignore unknown
                    {d_mu: 0, Dagger(d_mu): 0}
                ),
                factor,  # type: ignore unknown
            )
        except KeyError:
            continue
    return out  # type: ignore unknown


def simplify_inner_products(expr: sp.Expr) -> sp.Expr:
    return sp.expand(expr).subs(  # type: ignore unknown
        {  # States are normalized
            sp.Mul(Dagger(initial_state), initial_state): 1,  # type: ignore unknown
            sp.Mul(Dagger(state_perturbation), state_perturbation): 1,  # type: ignore unknown
            sp.Mul(Dagger(other_state), other_state): 1,  # type: ignore unknown
            sp.Mul(Dagger(orthogonal_state), orthogonal_state): 1,  # type: ignore unknown
            # The initial state is orthogonal to the perturbation
            sp.Mul(Dagger(initial_state), state_perturbation): 0,  # type: ignore unknown
            sp.Mul(Dagger(state_perturbation), initial_state): 0,  # type: ignore unknown
            # The other state is orthogonal to the initial
            sp.Mul(Dagger(other_state), initial_state): 0,  # type: ignore unknown
            sp.Mul(Dagger(initial_state), other_state): 0,  # type: ignore unknown
            # The orthogonal state is orthogonal to the initial
            sp.Mul(Dagger(orthogonal_state), initial_state): 0,  # type: ignore unknown
            sp.Mul(Dagger(initial_state), orthogonal_state): 0,  # type: ignore unknown
            # The orthogonal state is orthogonal to the other state
            sp.Mul(Dagger(orthogonal_state), other_state): 0,  # type: ignore unknown
            sp.Mul(Dagger(other_state), orthogonal_state): 0,  # type: ignore unknown
        }
    )


def _get_operator_expansion(operator: Operator) -> sp.Expr:
    out: sp.Expr = sp.Number(0)
    for i, state_i in enumerate([initial_state, state_perturbation]):
        for j, state_j in enumerate([initial_state, state_perturbation]):
            out += sp.Mul(  # type: ignore unknown
                sp.Symbol(f"{operator.label[0]}_{i}{j}"),  # type: ignore unknown
                state_i,
                Dagger(state_j),  # type: ignore unknown
            )
    return out  # type: ignore unknown


def express_as_c_number(expr: sp.Expr, operator: Operator) -> sp.Expr:
    expansion = _get_operator_expansion(operator)
    return sp.simplify(expr.subs({operator: expansion}))  # type: ignore unknown


def _get_operator_c_numbers(operator: Operator) -> dict[sp.Symbol, sp.Expr]:
    out = dict[sp.Symbol, sp.Expr]()
    for i, state_i in enumerate([initial_state, state_perturbation]):
        for j, state_j in enumerate([initial_state, state_perturbation]):
            symbol = sp.Symbol(f"{operator.label[0]}_{i}{j}")  # type: ignore unknown
            out[sp.conjugate(symbol)] = Dagger(state_j) * Dagger(operator) * state_i  # type: ignore unknown
            out[symbol] = Dagger(state_i) * operator * state_j  # type: ignore unknown
    return out


def express_as_inner_products(expr: sp.Expr, operator: Operator) -> sp.Expr:
    c_numbers = _get_operator_c_numbers(operator)
    return expr.subs(c_numbers)  # type: ignore unknown


def get_first_order_perturbation_initial_state() -> sp.Expr:
    # Evaluate the functional for the perturbed state
    functional_perturbed_first_order = extract_mu_factor(
        apply_m(perturbed_state, perturbed_mu),
        1,
    )
    # Find the component in the direction of the initial state
    out = sp.expand(sp.Mul(Dagger(initial_state), functional_perturbed_first_order))  # type: ignore unknown
    # Use initial eigenvalue equation to replace the hamiltonian dependent part
    out = out.subs(  # type: ignore unknown
        {
            -Dagger(simple_a_part * initial_state): -Dagger(  # type: ignore unknown
                initial_lambda * initial_state + _apply_drift_part(initial_state)  # type: ignore unknown
            )
        }
    )
    # Replace all inner products with the initial state
    out = simplify_inner_products(out)  # type: ignore unknown
    # Replace all inner products with their c numbers and simplify
    return express_as_inner_products(
        simplify_inner_products(express_as_c_number(out, L)), L
    )


def get_first_order_perturbation_other_state() -> sp.Expr:
    # Evaluate the functional for the perturbed state
    functional_perturbed_first_order = extract_mu_factor(
        apply_m(perturbed_state, perturbed_mu), 1
    )
    # Find the component in the direction of the initial state
    out = sp.expand(sp.Mul(Dagger(other_state), functional_perturbed_first_order))  # type: ignore unknown
    # Use initial eigenvalue equation to replace the hamiltonian dependent part
    out = out.subs(  # type: ignore unknown
        {
            -Dagger(simple_a_part * other_state): -Dagger(  # type: ignore unknown
                other_lambda * other_state + _apply_drift_part(other_state)  # type: ignore unknown
            )
        }
    )
    # Replace all inner products with the initial state
    return simplify_inner_products(out)


print("B first order terms")
print("Initial state")
sp.print_latex(get_first_order_perturbation_initial_state())
print("Other state")
sp.print_latex(get_first_order_perturbation_other_state())
