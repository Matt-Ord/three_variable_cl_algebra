from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sdeint  # type: ignore[import-untyped]
import sympy as sp
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

if TYPE_CHECKING:
    from .physical_systems import EtaParameters
hbar_value = 1.0545718e-34


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
            * KBT_value
            / hbar_value
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
                KBT: hbar,  # KBT = hbar, scaled
            }
        )
    )


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
def run_projected_simulation(config: SimulationConfig) -> SimulationResult:
    np.array([config.alpha_0, config.zeta_0], dtype=np.complex128)
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
    diff_expr = sp.Matrix([[alpha_derivative_diffusion, 0], [0, 0]])

    t = sp.Symbol("t", real=True)
    drift_func = sp.lambdify(
        (t, sp.Matrix([alpha, squeeze_ratio])), drift_expr, modules="numpy"
    )  # type: ignore[no-redef]
    diff_func = sp.lambdify(
        (t, sp.Matrix([alpha, squeeze_ratio])), diff_expr, modules="numpy"
    )  # type: ignore[no-redef]

    def coherent_derivative(
        y: np.ndarray[Any, np.dtype[np.complex128]], t: float
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.array(drift_func(t, y)).astype(np.complex128).flatten()  # type: ignore unknown

    def stochastic_derivative(
        y: np.ndarray[Any, np.dtype[np.complex128]], t: float
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.array(diff_func(t, y)).astype(np.complex128)  # type: ignore unknown

    # Get the simulation times and numerical noise
    simulation_times, numerical_noise = simulation_time_and_noise(config)
    # Run the simulation using the SDE solver
    sol = sdeint.stratSRS2(
        f=coherent_derivative,
        G=stochastic_derivative,
        y0=np.array([config.alpha_0, config.r_0], dtype=np.complex128),
        tspan=simulation_times,
        dW=numerical_noise,
    )
    return SimulationResult(
        times=config.times,
        params=config.params,
        alpha=sol[:, 0].astype(np.complex128),
        squeeze_ratio=sol[:, 1].astype(np.complex128),
        numerical_noise=numerical_noise[:, 0] + 1j * numerical_noise[:, 1],
    )
