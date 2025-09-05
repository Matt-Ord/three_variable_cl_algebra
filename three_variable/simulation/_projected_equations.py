from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sdeint  # type: ignore[import-untyped]
import sympy as sp
from scipy.constants import hbar as hbar_value  # type: ignore[import-untyped]
from slate_core.util import timed
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    expect_p,
    expect_x,
)
from three_variable.projected_sse import (
    get_deterministic_derivative,
    get_full_derivative,
    get_stochastic_derivative,
)
from three_variable.simulation.physical_systems import Units
from three_variable.symbols import (
    KBT,
    alpha,
    eta_lambda,
    eta_m,
    eta_omega,
    noise,
    zeta,
)

if TYPE_CHECKING:
    from three_variable.simulation.physical_systems import EtaParameters

from three_variable.equilibrium_squeeze import (
    get_equilibrium_squeeze_ratio,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)


@dataclass(frozen=True, kw_only=True)
class SimulationResult:
    params: EtaParameters
    times: np.ndarray[Any, np.dtype[np.float64]]
    alpha: np.ndarray[Any, np.dtype[np.complex128]]
    squeeze_ratio: np.ndarray[Any, np.dtype[np.complex128]]
    numerical_noise: np.ndarray[Any, np.dtype[np.complex128]]

    @property
    def x(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        expect_x_r = squeeze_ratio_from_zeta_expr(expect_x).subs(
            eta_m, self.params.eta_m
        )
        expect_x_fn = sp.lambdify((alpha, squeeze_ratio), expect_x_r, modules="numpy")
        return np.real(expect_x_fn(self.alpha, self.squeeze_ratio))  # type: ignore[unknown]

    @property
    def p(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        expect_p_r = squeeze_ratio_from_zeta_expr(expect_p).subs(
            eta_m, self.params.eta_m
        )
        expect_p_fn = sp.lambdify(
            (alpha, squeeze_ratio),
            expect_p_r.subs(hbar, hbar_value),
            modules="numpy",  # convert units
        )
        return np.real(expect_p_fn(self.alpha, self.squeeze_ratio))  # type: ignore[unknown]

    @property
    def alpha_full_derivative(self) -> np.ndarray[Any, np.dtype[np.complex128]]:
        """Calculate the full derivative of alpha with respect to time based on formula."""
        alpha_derivative = squeeze_ratio_from_zeta_expr(
            get_full_derivative("alpha").subs(sp.Symbol("V_1"), 0)
        )
        alpha_derivative = (
            explicit_from_dimensionless(alpha_derivative, self.params)
            * self.params.kbt_div_hbar
        )
        alpha_derivative_fn = sp.lambdify(
            (alpha, squeeze_ratio, noise), alpha_derivative, modules="numpy"
        )
        return np.array(
            alpha_derivative_fn(self.alpha, self.squeeze_ratio, self.numerical_noise)  # type: ignore[unknown]
        ).astype(np.complex128)  # type: ignore[unknown]

    @property
    def x_derivative_equilibrium(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate the equilibrium derivative of x from alpha full derivative."""
        expect_x_r0 = squeeze_ratio_from_zeta_expr(expect_x).subs(
            eta_m, self.params.eta_m
        )
        expect_dxdt_fn = sp.lambdify(
            (alpha, squeeze_ratio), expect_x_r0, modules="numpy"
        )
        return np.real(expect_dxdt_fn(self.alpha_full_derivative, self.squeeze_ratio))  # type: ignore[unknown]

    def __getitem__(self, key: slice) -> SimulationResult:
        return SimulationResult(
            times=self.times[key],
            alpha=self.alpha[key],
            squeeze_ratio=self.squeeze_ratio[key],
            params=self.params,
            numerical_noise=self.numerical_noise[key],  # type: ignore[no-redef]
        )


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    params: EtaParameters
    alpha_0: complex
    r_0: complex
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
                KBT: params.kbt_div_hbar * hbar,
            }
        )
    )


def evaluate_equilibrium_squeeze_ratio(
    eta_lambda_val: float, eta_omega_val: float
) -> np.complex128:
    """Estimate the initial value of r0 based on eta_lambda and eta_omega."""
    r_eq = get_equilibrium_squeeze_ratio()
    r0 = sp.lambdify((eta_lambda, eta_omega), r_eq, modules="numpy")  # type: ignore[no-redef]
    return r0(eta_lambda_val, eta_omega_val)  # type: ignore unknown


def _get_simulation_times(
    config: SimulationConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate simulation times and numerical noise."""
    # transform time to units of hbar / KBT
    return Units().time_into(config.times, config.params.units)


def _get_simulation_noise(
    n: int, dt: float
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Generate simulation times and numerical noise."""
    return sdeint.deltaW(n, 2, dt)


@timed
def run_projected_simulation(config: SimulationConfig) -> SimulationResult:
    params = config.params

    zeta_derivative = get_deterministic_derivative("zeta").subs(sp.Symbol("V_1"), 0)
    # transform zeta derivative to R0 derivative for numerical stability
    r_derivative = (
        squeeze_ratio_from_zeta_expr(-2 * zeta_derivative / (1 + zeta) ** 2) / eta_m
    )

    # write alpha derivatives in terms of R0
    alpha_derivative_deterministic = squeeze_ratio_from_zeta_expr(
        get_deterministic_derivative("alpha").subs(sp.Symbol("V_1"), 0)
    )
    alpha_derivative_diffusion_re = squeeze_ratio_from_zeta_expr(
        get_stochastic_derivative("alpha").subs(noise, 1)
    )
    alpha_derivative_diffusion_im = squeeze_ratio_from_zeta_expr(
        get_stochastic_derivative("alpha").subs(noise, 1j)
    )
    # substitute explicit values into the derivatives
    alpha_derivative_deterministic = explicit_from_dimensionless(
        alpha_derivative_deterministic, params
    )
    alpha_derivative_diffusion_re = explicit_from_dimensionless(
        alpha_derivative_diffusion_re, params
    )
    alpha_derivative_diffusion_im = explicit_from_dimensionless(
        alpha_derivative_diffusion_im, params
    )
    r_derivative_deterministic = explicit_from_dimensionless(r_derivative, params)
    # Generate a matrix equation for the drift and diffusion terms
    # And turn them into numpy ufuncs for the stochastic differential equation solver
    drift_expr = sp.Matrix([alpha_derivative_deterministic, r_derivative_deterministic])
    diff_expr = sp.Matrix(
        [[alpha_derivative_diffusion_re, alpha_derivative_diffusion_im], [0, 0]]
    )
    t = sp.Symbol("t", real=True)
    drift_func = sp.lambdify(
        (t, sp.Matrix([alpha, squeeze_ratio])),  # type: ignore[no-redef]
        drift_expr,  # type: ignore[no-redef]
        modules="numpy",
    )
    diff_func = sp.lambdify(
        (t, sp.Matrix([alpha, squeeze_ratio])),  # type: ignore[no-redef]
        diff_expr,  # type: ignore[no-redef]
        modules="numpy",
    )

    def coherent_derivative(
        y: np.ndarray[Any, np.dtype[np.complex128]], t: float
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.array(drift_func(t, y)).astype(np.complex128).flatten()  # type: ignore unknown

    def stochastic_derivative(
        y: np.ndarray[Any, np.dtype[np.complex128]], t: float
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.array(diff_func(t, y)).astype(np.complex128)  # type: ignore unknown

    simulation_times = _get_simulation_times(config)
    simulation_noise = _get_simulation_noise(
        len(simulation_times) - 1, simulation_times[1] - simulation_times[0]
    )
    # Run the simulation using the SDE solver
    sol = sdeint.stratSRS2(
        f=coherent_derivative,
        G=stochastic_derivative,
        y0=np.array([config.alpha_0, config.r_0], dtype=np.complex128),
        tspan=simulation_times,
        dW=simulation_noise,
    )
    return SimulationResult(
        times=config.times,
        params=config.params,
        alpha=sol[:, 0].astype(np.complex128),
        squeeze_ratio=sol[:, 1].astype(np.complex128),
        numerical_noise=simulation_noise[:, 0] + 1j * simulation_noise[:, 1],
    )
