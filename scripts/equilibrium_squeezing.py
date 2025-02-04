# What is the equilibrium value of
# the squeeze parameter beta?

from __future__ import annotations

import sympy as sp

# Define symbols
beta = sp.symbols("beta", complex=True)
hbar, m, lambda_, KBT, omega = sp.symbols(
    r"hbar m lambda KBT omega", real=True, positive=True
)
tau = 2 * sp.sqrt(KBT * m / (hbar**2))

beta_star = sp.conjugate(beta)

# Define the equation
expr = (
    (sp.I * hbar / (2 * m)) * (1 - beta_star) ** 2
    - (sp.I * m * omega**2 / (2 * hbar)) * (1 + beta_star) ** 2
    + (lambda_ / (2 * tau**2)) * (tau**2 * (1 + beta_star) + (1 - beta_star)) ** 2
)


# sp.print_latex(sp.Poly(expr, beta_star).as_expr())
# equation = sp.Eq(sp.Poly(expr, beta_star).as_expr(), 0)
# solution = sp.solve(equation, beta)
# print("naiive solution")
# sp.print_latex(sp.simplify(solution[0]))
# Set the equation to zero
equation = sp.Poly(expr, beta_star)

# Display the equation
a, b, c = equation.all_coeffs()
quadratic_solution = (
    (-b + sp.sqrt(sp.simplify(b**2 - 4 * a * c))) / (2 * a),
    (-b - sp.sqrt(sp.simplify(b**2 - 4 * a * c))) / (2 * a),
)


# solution = sp.solve(equation, beta_star)
print("expresssion")
sp.print_latex(expr)
print("equation")
sp.print_latex(equation.as_expr())
print("a")
sp.print_latex(sp.simplify(a))
print("b")
sp.print_latex(sp.simplify(b))
print("c")
sp.print_latex(sp.simplify(c))
print()
print("answer")
answer = sp.simplify(quadratic_solution[0]).subs(
    KBT, sp.symbols("K_b") * sp.symbols("T")
)
sp.print_latex(answer)
print()
# sp.print_latex(sp.simplify(answer * sp.conjugate(answer)))


print("simplified")

beta_0 = (hbar - m * omega) / (hbar + m * omega)
shifted_beta = (beta_0 + beta_star) / (1 + beta_star * beta_0)


expr_subbed = expr.subs(beta_star, shifted_beta)
print("shifted expresssion")
# This is the shifted beta consistency equation
sp.print_latex(sp.factor_terms(sp.simplify(expr_subbed)))

eta = sp.symbols("eta", real=True)

shifted_expr = (
    -(16 * sp.I * omega) * (beta_star) * eta
    + (lambda_) * (4 * eta * (1 + beta_star) + (1 - beta_star)) ** 2
)
shifted_equation = sp.Poly(shifted_expr, beta_star)

# Display the equation
a, b, c = shifted_equation.all_coeffs()
quadratic_solution = (
    (-b + sp.sqrt(sp.simplify(b**2 - 4 * a * c))) / (2 * a),
    (-b - sp.sqrt(sp.simplify(b**2 - 4 * a * c))) / (2 * a),
)
print()
print("answer")
answer = sp.simplify(quadratic_solution[0]).subs(
    {KBT: sp.symbols("K_b") * sp.symbols("T")}
)
sp.print_latex(answer)

beta_0 = (1 - 4 * eta) / (1 + 4 * eta)
shifted_subbed_beta = beta_0 + beta_star  # / (1 + beta_star * beta_0)
expr_subbed = shifted_expr.subs(beta_star, shifted_subbed_beta)
print("shifted expresssion subbed")
# This is the shifted beta delta consistency equation
sp.print_latex(
    sp.collect(sp.factor_terms(sp.simplify(sp.expand(expr_subbed))), beta_star)
)


eta = KBT / (hbar * omega)
beta_0 = (hbar - m * omega) / (hbar + m * omega)
beta_1 = (1 - 4 * eta) / (1 + 4 * eta)
beta_0_1 = (beta_0 + beta_1) / (1 + beta_0 * beta_1)
shifted_beta_0_1 = (beta_0_1 + beta_star) / (1 + beta_star * beta_0_1)
shifted_v2 = expr.subs(beta_star, shifted_beta_0_1)
print("twice shifted expresssion")
sp.print_latex(sp.collect(sp.factor_terms(sp.simplify(shifted_v2)), beta_star))
