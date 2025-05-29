# Can we find the coherent states which minimuise the average drift?
from __future__ import annotations

import sympy as sp

from three_variable.equilibrium_squeeze import (
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import get_full_derivative

derivative = squeeze_ratio_from_zeta_expr(get_full_derivative("zeta"))
sp.print_latex(derivative)
sp.print_latex(sp.solve(derivative, squeeze_ratio)[0])
