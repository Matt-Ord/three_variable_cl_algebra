# What factors of x should we use to expand the potential energy term?
# We want x powers that only contribute to nth order squeezing terms.
from __future__ import annotations

from functools import cache

import sympy as sp
from sympy.physics.secondquant import (
    CreateBoson,
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

phi = sp.symbols(("phi"), real=True)
alpha = sp.symbols(r"\alpha", complex=True)
mu = sp.symbols(r"\mu", real=True)
nu = sp.symbols(r"\nu", real=False)


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


x_operators = [get_x_operator(0) ** n for n in range(4)]

shift = (alpha * (mu + sp.conjugate(nu)) + sp.conjugate(alpha) * (nu + mu)) / sp.sqrt(2)
shifted_x = sp.simplify(get_x_operator(0) - shift)
shifted_x = [(shifted_x) ** n for n in range(4)]

correction = sp.simplify((mu + nu) * (mu + sp.conjugate(nu)))
sp.print_latex(sp.simplify(get_x_operator(0) - shift))

for i in range(4):
    state = FockStateBosonKet([i])
    vaccum = FockStateBosonKet([0])
    print("--------------------------------------")
    print(f"n={i}")
    sp.print_latex(
        sp.collect(
            sp.factor_terms(apply_operators(Dagger(state) * shifted_x[0] * vaccum)),
            [alpha, sp.conjugate(alpha)],
        )
    )
    sp.print_latex(
        sp.collect(
            sp.factor_terms(apply_operators(Dagger(state) * (shifted_x[1]) * vaccum)),
            [alpha, sp.conjugate(alpha)],
        )
    )

    # corrected_x = sp.simplify(shifted_x[2] - (correction / 2))
    corrected_x = shifted_x[2] - (correction / 2)
    sp.print_latex(
        sp.collect(
            sp.factor_terms(apply_operators(Dagger(state) * corrected_x * vaccum)),
            [alpha, sp.conjugate(alpha)],
        )
    )
    # sp.print_latex(
    #     sp.collect(
    #         sp.factor_terms(apply_operators(Dagger(state) * shifted_x[3] * vaccum)),
    #         [alpha, sp.conjugate(alpha)],
    #     )
    # )
