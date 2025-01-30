from __future__ import annotations

import sympy as sp

# Define symbolic variables
alpha, beta = sp.symbols(r"\alpha \beta", complex=True)
M_inv = sp.Matrix(
    [[sp.I, sp.conjugate(alpha) / 2, -sp.conjugate(beta) / 2], [0, 1, 0], [0, 0, 1]]
)
M = M_inv.inv()
sp.pprint(M)
