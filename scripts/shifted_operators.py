# Attempt to answer the question - what does an nth squeeze operator look like?
from __future__ import annotations

import sympy as sp
from sympy.physics.secondquant import (
    AnnihilateBoson,
    CreateBoson,
    Dagger,
    FockStateBosonKet,
    apply_operators,
)


def _normal_order_mul(m: sp.Mul) -> sp.Expr:
    c_part, nc_part = m.args_cnc()

    ordered_nc_part = list[sp.Expr]()
    annihilators = list[sp.Expr]()
    for nc in nc_part:
        if isinstance(nc, CreateBoson):
            ordered_nc_part.append(nc)
            continue
        if isinstance(nc, sp.Pow) and isinstance(nc.base, CreateBoson):
            ordered_nc_part.append(nc)
            continue
        if isinstance(nc, AnnihilateBoson):
            annihilators.append(nc)
            continue
        if isinstance(nc, sp.Pow) and isinstance(nc.base, AnnihilateBoson):
            annihilators.append(nc)
            continue

        msg = f"Cannot handle {nc}"
        raise NotImplementedError(msg)

    return sp.Mul(*c_part) * sp.Mul(*ordered_nc_part, *annihilators)


def normal_order_operators(e: sp.Expr) -> sp.Expr:
    """Given an expression, order the operators in normal order."""
    e = e.expand()
    muls = e.atoms(sp.Mul)
    subs_list = [(m, _normal_order_mul(m)) for m in iter(muls)]
    return e.subs(subs_list)


def get_nth_expectation(n: int, shifted_a: sp.Expr) -> sp.Expr:
    if n == 0:
        return sp.Integer(1)

    vaccum = FockStateBosonKet([0])
    if n == 1:
        return apply_operators(Dagger(vaccum) * shifted_a * vaccum)
    return apply_operators(
        Dagger(vaccum) * (shifted_a - get_nth_expectation(1, shifted_a)) ** n * vaccum
    )


def get_x_operator(shifted_a: sp.Expr) -> sp.Expr:
    return (shifted_a + Dagger(shifted_a)) / sp.sqrt(2)


def get_nth_x_expectation(n: int, shifted_create: sp.Expr) -> sp.Expr:
    if n == 0:
        return sp.Integer(1)

    vaccum = FockStateBosonKet([0])
    shifted_x = get_x_operator(shifted_create)
    if n == 1:
        return apply_operators(Dagger(vaccum) * shifted_x * vaccum)
    expect_x = apply_operators(Dagger(vaccum) * shifted_x * vaccum)
    dummy_a = CreateBoson(0)
    dummy_x = get_x_operator(dummy_a)
    normal_ordered_expect = normal_order_operators((dummy_x - expect_x) ** n)

    normal_ordered_expect = normal_ordered_expect.subs(
        {dummy_a: shifted_create, Dagger(dummy_a): Dagger(shifted_create)}
    )
    return apply_operators(Dagger(vaccum) * normal_ordered_expect * vaccum)


get_nth_x_expectation(4, CreateBoson(0))

alpha = sp.Symbol("alpha")
shifted_order_0 = CreateBoson(0) + alpha
beta = sp.Symbol("beta")
r = sp.Symbol("r", real=True)
theta = sp.Symbol("theta", real=True)
shifted_order_1 = sp.cosh(r) * CreateBoson(0) + sp.sinh(r) * sp.exp(
    sp.I * theta
) * Dagger(CreateBoson(0))
# shifted_order_1 /= sp.sqrt(1 + sp.Abs(beta) ** 2)
gamma = sp.Symbol("gamma")
gamma_1 = sp.Symbol("gamma_1")
shifted_order_2 = (
    CreateBoson(0)
    + alpha
    # + beta * Dagger(CreateBoson(0))
    # + gamma * Dagger(CreateBoson(0)) ** 2
    + gamma_1 * Dagger(CreateBoson(0)) * CreateBoson(0)
)

# for i in range(5):
#     print(f"Expectation for i = {i}")
#     print("Order 0")
#     sp.print_latex(get_nth_expectation(i, shifted_order_0))
#     print("Order 1")
#     sp.print_latex(get_nth_expectation(i, shifted_order_1))
#     print("Order 2")
#     sp.print_latex(get_nth_expectation(i, shifted_order_2))
#     print()

for i in range(5):
    print(f"Expectation for i = {i}")
    print("Standard")
    sp.print_latex(get_nth_x_expectation(i, CreateBoson(0)))
    print("Order 0")
    sp.print_latex(get_nth_x_expectation(i, shifted_order_0))
    print("Order 1")
    order_1 = get_nth_x_expectation(i, shifted_order_1)
    order_1 = sp.expand(
        order_1.subs(
            {
                sp.sinh(r) ** 5: sp.sinh(r) * (sp.cosh(r) ** 2 - 1) ** 2,
                sp.sinh(r) ** 3: sp.sinh(r) * (sp.cosh(r) ** 2 - 1),
                sp.sinh(r) ** 2: sp.cosh(r) ** 2 - 1,
            }
        )
    )
    sp.print_latex(sp.Poly(order_1, sp.cosh(r)).as_expr())
    # print("Order 2")
    # sp.print_latex(get_nth_x_expectation(i, shifted_order_2))
    print()
