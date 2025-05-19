from __future__ import annotations

import pytest
import sympy as sp

from three_variable.new_paper.projected_sse import (
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
    zeta_from_squeeze_ratio_expr,
)
from three_variable.new_paper.symbols import (
    dimensionless_from_full,
    full_from_dimensionless,
    lambda_,
    m,
    omega,
    zeta,
)


@pytest.mark.parametrize(
    ("symbol"),
    [(m), (omega), (lambda_)],
)
def test_conversion_dimensionless(symbol: sp.Symbol) -> None:
    # The conversion between dimensionless and dimensionful variables must be consistent

    assert (
        sp.simplify(symbol - full_from_dimensionless(dimensionless_from_full(symbol)))
        == 0
    )


def test_conversion_ratio() -> None:
    assert (
        sp.simplify(
            squeeze_ratio
            - squeeze_ratio_from_zeta_expr(zeta_from_squeeze_ratio_expr(squeeze_ratio))
        )
        == 0
    )
    assert (
        sp.simplify(
            zeta - zeta_from_squeeze_ratio_expr(squeeze_ratio_from_zeta_expr(zeta))
        )
        == 0
    )
