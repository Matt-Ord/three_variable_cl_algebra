from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.constants import angstrom  # type: ignore sp
from slate_core import plot

from three_variable.equilibrium_squeeze import evaluate_equilibrium_expect_x_squared
from three_variable.physical_systems import (
    ELENA_NA_CU,
    ELENA_NA_CU_BALLISTIC,
    BaseParameters,
)


def get_sigma_0(params: BaseParameters) -> float:
    eta = params.eta_parameters
    # Equal to sqrt( expect(x) - expect(x)^2 )
    return np.sqrt(
        2
        * evaluate_equilibrium_expect_x_squared(
            np.array([eta.eta_m]),  # type: ignore sp
            np.array([eta.eta_omega]),  # type: ignore sp
            np.array([eta.eta_lambda]),  # type: ignore sp
        ).item(0)
    )


def assert_elena_sigma_0() -> None:
    x = sp.Symbol("x", real=True)
    sigma_0 = sp.Symbol("sigma_0", real=True, positive=True)
    wavefunction = sp.exp(-(sp.Abs(x) ** 2) / (2 * sigma_0**2))
    norm = sp.integrate(sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))  # type: ignore sp
    wavefunction /= sp.sqrt(norm)  # type: ignore sp
    uncertainty_x = sp.integrate(x**2 * sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))  # type: ignore sp
    # sigma_0 = sqrt(2 * uncertainty_x) as we use above...
    assert sp.simplify(uncertainty_x - (sigma_0**2 / 2)) == 0  # type: ignore sp


def assert_periodic_width() -> None:
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True)
    sigma_0 = sp.Symbol("sigma_0", real=True, positive=True)
    wavefunction = sp.exp(-(sp.Abs(x) ** 2) / (2 * sigma_0**2))
    norm = sp.integrate(sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))  # type: ignore sp
    wavefunction /= sp.sqrt(norm)  # type: ignore sp
    periodic_x = sp.integrate(  # type: ignore sp
        sp.exp(sp.I * k * x) * sp.Abs(wavefunction) ** 2,  # type: ignore sp
        (x, -sp.oo, sp.oo),
    )
    sp.print_latex(sp.simplify(periodic_x))  # type: ignore sp

    x_squared = sp.integrate(x**2 * sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))  # type: ignore sp
    sp.print_latex(sp.simplify(x_squared))  # type: ignore sp


assert_periodic_width()


def get_sigma_0_coherent(params: BaseParameters) -> float:
    return np.sqrt(2 * params.eta_parameters.eta_omega / params.eta_parameters.eta_m)


def get_sigma_0_free(params: BaseParameters) -> float:
    return np.sqrt(2 * (np.sqrt(2) - 1) / 4 * params.eta_parameters.eta_m)


if __name__ == "__main__":
    low_temperatures = [10, 50, 100, 200, 300, 400, 500, 600, 700, 900]
    high_temperatures = [1000, 1200, 1500, 1700, 2000, 3000, 4000, 5000, 6000]
    temperatures = low_temperatures + high_temperatures

    params = ELENA_NA_CU.base_parameters
    data = [params.with_temperature(temperature) for temperature in temperatures]
    widths_sse_formula = [get_sigma_0(params) for params in data]
    print([f"{w:0.2e}" for w in widths_sse_formula])

    widths_coherent = [get_sigma_0_coherent(params) for params in data]
    print([f"{w:0.2e}" for w in widths_coherent])
    widths_free = [get_sigma_0_free(params) for params in data]
    print([f"{w:0.2e}" for w in widths_free])

    params = ELENA_NA_CU_BALLISTIC.base_parameters
    data = [params.with_temperature(temperature) for temperature in temperatures]
    widths_ballistic_formula = [get_sigma_0(params) for params in data]
    print([f"{w:0.2e}" for w in widths_ballistic_formula])

    fig, ax = plot.get_figure()
    (line,) = ax.plot(temperatures, widths_sse_formula)
    line.set_label("SSE Formula")
    (line,) = ax.plot(temperatures, widths_ballistic_formula)
    line.set_label("Ballistic Formula")
    (line,) = ax.plot(temperatures, widths_coherent)
    line.set_label("Coherent Widths")
    ax.set_ylim(0, None)

    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Width / m")
    ax.set_title("Width of the sodium copper system against temperature")
    ax.legend()
    fig.show()

    plot.wait_for_close()

    print(f"{3.615 * angstrom:0.2e}")
