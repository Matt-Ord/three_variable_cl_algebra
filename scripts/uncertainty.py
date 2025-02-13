from __future__ import annotations

import sympy as sp

from three_variable.equilibrium_squeeze import (
    R,
    get_equilibrium_squeeze_R,
    get_uncertainty_beta,
    get_uncertainty_p_R,
    get_uncertainty_R,
    get_uncertainty_x_beta,
    get_uncertainty_x_R,
)

print("Uncertainty x beta")
sp.print_latex(get_uncertainty_x_beta())

print("Uncertainty beta")
sp.print_latex(get_uncertainty_beta())
print()


print("Uncertainty R")
sp.print_latex(sp.Symbol(r"hbar") ** 2 * get_uncertainty_R())
print()
print("Uncertainty X R")
sp.print_latex(get_uncertainty_x_R())
print()
print("Uncertainty P R")
sp.print_latex(sp.Symbol(r"hbar") ** 2 * get_uncertainty_p_R())
print()

neum, denom = sp.together(get_uncertainty_R()).as_numer_denom()

equilibrium_R = get_equilibrium_squeeze_R()
neum_R = neum.subs({R: equilibrium_R})
neum_R = sp.simplify(sp.expand(neum_R))
denom_R = denom.subs({R: equilibrium_R})
denom_R = sp.simplify(sp.expand(denom_R))

sp.print_latex(neum_R)
print()
sp.print_latex(denom_R)
print()
