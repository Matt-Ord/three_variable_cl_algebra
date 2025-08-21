from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sdeint  # type: ignore[import-untyped]
import sympy as sp
<<<<<<< HEAD
=======
from slate_core.util import timed
>>>>>>> 2cca05c (analytical expression for gaussian solution to fokker planck equation, calculate energy distribution from analytical expression and from numerical simulation based on sde)
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    expect_p,
    expect_x,
)
from three_variable.projected_sse import (
    get_full_derivative,
    get_stochastic_derivative,
)
from three_variable.symbols import (
    KBT,
    alpha,
    eta_lambda,
    eta_m,
    eta_omega,
    noise,
    zeta,
)
from three_variable.symbols import (
    hbar as hbar_symbol,
)

<<<<<<< HEAD
if TYPE_CHECKING:
    from .physical_systems import EtaParameters
=======
hbar_value = 1.0545718e-34
>>>>>>> 2cca05c (analytical expression for gaussian solution to fokker planck equation, calculate energy distribution from analytical expression and from numerical simulation based on sde)


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


def explicit_from_dimensionless(expr: sp.Expr, params: EtaParameters) -> sp.Expr:
    """Convert a dimensionless value to explicit units."""
    return sp.simplify(
        expr.subs(
            {
                sp.Symbol("V_1"): 0,
                eta_lambda: params.eta_lambda,
                eta_m: params.eta_m,
                eta_omega: params.eta_omega,
                KBT: hbar,  # KBT = hbar, scaled
            }
        )
    )


<<<<<<< HEAD
=======
def estimate_r0(eta_lambda_val: float, eta_omega_val: float) -> np.complex128:
    """Estimate the initial value of r0 based on eta_lambda and eta_omega."""
    r_eq = get_equilibrium_squeeze_ratio()
    r0 = sp.lambdify((eta_lambda, eta_omega), r_eq, modules="numpy")  # type: ignore[no-redef]
    return r0(eta_lambda_val, eta_omega_val)  # type: ignore unknown


def simulation_time_and_noise(
    config: SimulationConfig,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """Generate simulation times and numerical noise."""
    # Generate the noise
    generator = np.random.default_rng(seed=42)
    # transform time to units of hbar / KBT
    simulation_times = config.times * config.params.kbt_div_hbar
    time_step = (simulation_times[len(simulation_times) - 1] - simulation_times[0]) / (
        len(simulation_times) - 1
    )  # assuming equal time steps
    numerical_noise = sdeint.deltaW(
        len(simulation_times) - 1, 2, time_step, generator
    )  # shape (N, 2)
    return simulation_times, numerical_noise


@timed
>>>>>>> 2cca05c (analytical expression for gaussian solution to fokker planck equation, calculate energy distribution from analytical expression and from numerical simulation based on sde)
def run_projected_simulation(config: SimulationConfig) -> SimulationResult:
    y0 = np.array([config.alpha_0, config.zeta_0], dtype=np.complex128)
    params = config.params

    alpha_derivative_deterministic = explicit_from_dimensionless(
        get_full_derivative("alpha"), params
    ).subs(noise, 0)

    alpha_derivative_diffusion = explicit_from_dimensionless(
        get_stochastic_derivative("alpha"), params
    ).subs(noise, 1)

    zeta_derivative_deterministic = explicit_from_dimensionless(
        get_full_derivative("zeta"), params
    ).subs(noise, 0)

    # Generate a matrix equation for the drift and diffusion terms
    # And turn them into numpy ufuncs for the stochastic differential equation solver
    drift_expr = sp.Matrix(
        [alpha_derivative_deterministic, zeta_derivative_deterministic]
    )
    diff_expr = sp.Matrix([[alpha_derivative_diffusion, 0], [0, 0]])

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
