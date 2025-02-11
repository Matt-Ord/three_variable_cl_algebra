from __future__ import annotations

import sympy as sp
from sympy.physics.secondquant import (
    Dagger,
    FockStateBosonKet,
    apply_operators,
)

from three_variable.equilibrium_squeeze import (
    get_uncertainty_p_beta,
    get_uncertainty_x_beta,
)
from three_variable.projected_sse import get_p_operator, get_x_operator
from three_variable.symbols import beta, hbar, mu, nu, phi


def test_uncertainty_x() -> None:
    uncertainty_x = get_uncertainty_x_beta()
    uncertainty_p = get_uncertainty_p_beta()

    vaccum = FockStateBosonKet([0])
    x_operator = get_x_operator(0)
    expect_x = apply_operators(Dagger(vaccum) * x_operator * vaccum)
    uncertainty_x_manual = (
        apply_operators(Dagger(vaccum) * (x_operator) * Dagger(x_operator) * vaccum)
        - expect_x**2
    )
    uncertainty_x_manual = sp.simplify(uncertainty_x_manual)

    uncertainty_x_manual = uncertainty_x_manual.subs({phi: 0, nu: beta * mu})
    uncertainty_x_manual = uncertainty_x_manual.subs(
        {
            mu: 1 / sp.sqrt(1 - sp.Abs(beta) ** 2),
            beta * sp.conjugate(beta): sp.Abs(beta) ** 2,
        }
    )

    assert sp.simplify(uncertainty_x_manual - 2 * uncertainty_x) == 0

    print("Uncertainty p")
    vaccum = FockStateBosonKet([0])
    p_operator = get_p_operator(0)
    expect_p = apply_operators(Dagger(vaccum) * p_operator * vaccum)
    uncertainty_p_manual = (
        apply_operators(Dagger(vaccum) * (p_operator) ** 2 * vaccum) - expect_p**2
    )
    uncertainty_p_manual = sp.simplify(uncertainty_p_manual)

    uncertainty_p_manual = uncertainty_p_manual.subs({phi: 0, nu: beta * mu})
    uncertainty_p_manual = uncertainty_p_manual.subs(
        {
            mu: 1 / sp.sqrt(1 - sp.Abs(beta) ** 2),
            beta * sp.conjugate(beta): sp.Abs(beta) ** 2,
        }
    )

    assert sp.simplify(uncertainty_p_manual - 2 * hbar**2 * uncertainty_p) == 0
