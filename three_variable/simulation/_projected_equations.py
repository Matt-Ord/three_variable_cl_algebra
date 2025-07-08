from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sdeint  # type: ignore[import-untyped]
import sympy as sp
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    action_from_expr,
    expect_p,
    expect_x,
    extract_action,
)
from three_variable.projected_sse import (
    get_diffusion_term,
    get_environment_derivative,
    get_full_derivative,
    get_system_derivative,
)
from three_variable.symbols import (
    KBT,
    alpha,
    dimensionless_from_full,
    eta_lambda,
    eta_m,
    eta_omega,
    noise,
    zeta,
)
from three_variable.symbols import (
    hbar as hbar_symbol,
)

if TYPE_CHECKING:
    from .physical_systems import EtaParameters


@dataclass(frozen=True, kw_only=True)
class SimulationResult:
    params: EtaParameters
    times: np.ndarray[Any, np.dtype[np.float64]]
    alpha: np.ndarray[Any, np.dtype[np.complex128]]
    zeta: np.ndarray[Any, np.dtype[np.complex128]]

    @property
    def x(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        expect_x_fn = sp.lambdify((alpha, zeta), expect_x, modules="numpy")
        return np.real(expect_x_fn(self.alpha, self.zeta))  # type: ignore[unknown]

    @property
    def p(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        expect_p_fn = sp.lambdify(
            (alpha, zeta), expect_p.subs(hbar_symbol, 1), modules="numpy"
        )
        return np.real(expect_p_fn(self.alpha, self.zeta))  # type: ignore[unknown]

    def __getitem__(self, key: slice) -> SimulationResult:
        return SimulationResult(
            times=self.times[key],
            alpha=self.alpha[key],
            zeta=self.zeta[key],
            params=self.params,
        )


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    params: EtaParameters
    alpha_0: complex
    zeta_0: complex
    times: np.ndarray[Any, np.dtype[np.float64]]


def run_projected_simulation(config: SimulationConfig) -> SimulationResult:
    y0 = np.array([config.alpha_0, config.zeta_0], dtype=np.complex128)
    params = config.params

    alpha_derivative_deterministic = get_full_derivative("alpha")
    alpha_derivative_deterministic = sp.simplify(
        alpha_derivative_deterministic.subs(
            {
                sp.Symbol("V_1"): 0,
                eta_lambda: params.eta_lambda,
                eta_m: params.eta_m,
                eta_omega: params.eta_omega,
                KBT: params.kbt_div_hbar * hbar,
            }
        )
    )

    alpha_derivative_diffusion = get_diffusion_term()
    alpha_derivative_diffusion = sp.factor(
        extract_action(action_from_expr(alpha_derivative_diffusion), "alpha")
    )
    alpha_derivative_diffusion = sp.simplify(
        dimensionless_from_full(alpha_derivative_diffusion),
        rational=True,
    )
    alpha_derivative_diffusion = sp.simplify(
        alpha_derivative_diffusion.subs(
            {
                sp.Symbol("V_1"): 0,
                eta_lambda: params.eta_lambda,
                eta_m: params.eta_m,
                eta_omega: params.eta_omega,
                KBT: params.kbt_div_hbar * hbar,
            }
        )
    )

    expr_system = get_system_derivative("zeta")
    expr_environment = get_environment_derivative("zeta")

    zeta_derivative = expr_system + expr_environment
    zeta_derivative = sp.simplify(
        zeta_derivative.subs(
            {
                sp.Symbol("V_1"): 0,
                eta_lambda: params.eta_lambda,
                eta_m: params.eta_m,
                eta_omega: params.eta_omega,
                KBT: params.kbt_div_hbar * hbar,
            }
        )
    )

    # Generate a matrix equation for the drift and diffusion terms
    # And turn them into numpy ufuncs for the stochastic differential equation solver
    drift_expr = sp.Matrix([alpha_derivative_deterministic, zeta_derivative])
    diff_expr = sp.Matrix([[alpha_derivative_diffusion.subs(noise, 1), 0], [0, 0]])

    t = sp.Symbol("t", real=True)
    drift_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), drift_expr, modules="numpy")  # type: ignore[no-redef]
    diff_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), diff_expr, modules="numpy")  # type: ignore[no-redef]

    def coherent_derivative(
        y: np.ndarray[Any, np.dtype[np.complex128]], t: float
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.array(drift_func(t, y)).astype(np.complex128).flatten()  # type: ignore unknown

    def stochastic_derivative(
        y: np.ndarray[Any, np.dtype[np.complex128]], t: float
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.array(diff_func(t, y)).astype(np.complex128)  # type: ignore unknown

    sol = sdeint.itoint(coherent_derivative, stochastic_derivative, y0, config.times)
    return SimulationResult(
        times=config.times,
        params=config.params,
        alpha=sol[:, 0].astype(np.complex128),
        zeta=sol[:, 1].astype(np.complex128),
    )
