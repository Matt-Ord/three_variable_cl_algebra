from __future__ import annotations

import sympy as sp

from three_variable.symbols import (
    KBT,
    eta_lambda_expression,
    eta_m,
    eta_m_expression,
    eta_omega_expression,
    hbar,
    lambda_,
    m,
    omega,
)

beta_star = sp.conjugate(beta)

derivative_expression = (
    (sp.I * hbar / (2 * m)) * (1 - beta_star) ** 2
    - (sp.I * m * omega**2 / (2 * hbar)) * (1 + beta_star) ** 2
    + (lambda_ / (4 * eta_m))
    * (
        (1 - beta_star) ** 2
        - 4 * eta_m * (1 - beta_star) * (1 + beta_star)
        - 4 * eta_m * (1 + beta_star) ** 2
    )
)


derivative_subbed = derivative_expression.subs(
    {m: eta_m_expression, lambda_: eta_lambda_expression, omega: eta_omega_expression}
)
factor = KBT / (hbar * eta_m)
# Check that the eta form of the derivative is correct
sp.print_latex(sp.factor_terms(sp.simplify(derivative_subbed), factor))
