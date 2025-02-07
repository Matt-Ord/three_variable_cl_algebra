from __future__ import annotations

import sympy as sp

from three_variable.equilibrium_squeeze import (
    eta_m,
    eta_omega,
    get_equilibrium_squeeze_beta,
    get_equilibrium_squeeze_derivative,
    get_equilibrium_squeeze_derivative_gradient,
    get_equilibrium_squeeze_R,
)
from three_variable.projected_sse import (
    get_squeeze_derivative_environment_beta,
    get_squeeze_derivative_system_beta,
)
from three_variable.symbols import beta


def factorize_derivative_lambda(derivative: sp.Expr) -> sp.Expr:
    print("factorize_derivative_lambda")
    derivative_lambda_eta_fraction = (2 * eta_m) / (sp.sqrt(2) - 1)
    beta_lambda_expression = (1 - derivative_lambda_eta_fraction) / (
        1 + derivative_lambda_eta_fraction
    )

    squeeze_derivative_lambda = get_squeeze_derivative_environment_beta()
    derivative_lambda_solution = sp.Subs(
        squeeze_derivative_lambda, sp.conjugate(beta), beta_lambda_expression
    )
    assert sp.factor(sp.simplify(sp.expand(derivative_lambda_solution))) == 0

    # Get an equation only involving beta_1
    # Using beta = beta_0 + beta_1 / (1 + beta_0 * beta_1)
    # and the anzatz beta_0 = (1 + 2 eta_m) / (1 - 2 eta_m) chosen to simplify the
    # last term in the derivative
    beta_0 = sp.symbols("beta_0")
    beta_1 = sp.symbols("beta_1")

    equilibrium_beta = (beta_0 + beta_1) / (1 + beta_0 * beta_1)
    derivative_subbed = derivative.subs(
        {beta: equilibrium_beta.subs({beta_0: beta_lambda_expression})}
    )
    derivative_subbed = sp.together(derivative_subbed)
    sp.print_latex(sp.together(derivative_subbed))
    print()

    # Drop the denominator, since when the numerator is zero, the fraction is zero
    numerator, _ = derivative_subbed.as_numer_denom()
    beta_1_equation = sp.factor_terms(
        sp.Poly(numerator, sp.conjugate(beta_1)).as_expr()
    )
    # Here we can see that the expression splits into a part factorized by (1-beta_1)^2 and
    # one part that is proportional to beta_1^2
    sp.print_latex(beta_1_equation)
    print()


def factorize_derivative_mass(derivative: sp.Expr) -> sp.Expr:
    print("factorize_derivative_mass")

    beta_m_expression = (1 - (eta_m / eta_omega)) / (1 + (eta_m / eta_omega))

    squeeze_derivative_mass = get_squeeze_derivative_system_beta()
    derivative_mass_solution = sp.Subs(
        squeeze_derivative_mass, sp.conjugate(beta), beta_m_expression
    )
    assert sp.factor(sp.simplify(sp.expand(derivative_mass_solution))) == 0
    beta_0 = sp.symbols("beta_0")
    beta_1 = sp.symbols("beta_1")

    equilibrium_beta = (beta_0 + beta_1) / (1 + beta_0 * beta_1)
    derivative_subbed = derivative.subs(
        {beta: equilibrium_beta.subs({beta_0: beta_m_expression})}
    )
    sp.print_latex(sp.together(derivative_subbed))
    print()
    numerator, _ = derivative_subbed.as_numer_denom()
    beta_1_equation = sp.factor_terms(
        sp.Poly(numerator, sp.conjugate(beta_1)).as_expr()
    )
    # Here we can see that the expression splits into a part factorized by (1-beta_1)^2 and
    # one part that is proportional to beta_1^2
    sp.print_latex(beta_1_equation)
    print()
    input()


print("get_equilibrium_squeeze_R")
sp.print_latex(get_equilibrium_squeeze_R())
print()
print("get_equilibrium_squeeze_beta")
sp.print_latex(get_equilibrium_squeeze_beta())
print()

assert get_equilibrium_squeeze_derivative() == 0

sp.print_latex(get_equilibrium_squeeze_derivative_gradient())
