from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.constants import angstrom

from three_variable import util
from three_variable.equilibrium_squeeze import (
    R,
    evaluate_equilibrium_R,
    get_uncertainty_x_R,
)
from three_variable.physical_systems import (
    ELENA_NA_CU,
    ELENA_NA_CU_BALLISTIC,
    BaseParameters,
)


def evaluate_equilibrium_uncertainty_x(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    equilibrium_R = evaluate_equilibrium_R(  # noqa: N806
        eta_m, eta_omega, eta_lambda, positive=positive
    )
    # sp.print_latex(get_equilibrium_squeeze_R(positive=positive))
    # sp.print_latex(get_uncertainty_x_R())
    uncertainty_from_R = sp.lambdify((R), get_uncertainty_x_R())
    return np.real_if_close(uncertainty_from_R(equilibrium_R))


def get_sigma_0(params: BaseParameters) -> float:
    eta = params.eta_parameters
    # Equal to sqrt( expect(x) - expect(x)^2 )
    return np.sqrt(
        2
        * evaluate_equilibrium_uncertainty_x(
            np.array([eta.eta_m]),
            np.array([eta.eta_omega]),
            np.array([eta.eta_lambda]),
        ).item(0)
    )


def assert_elena_sigma_0() -> None:
    x = sp.Symbol("x", real=True)
    sigma_0 = sp.Symbol("sigma_0", real=True, positive=True)
    wavefunction = sp.exp(-(sp.Abs(x) ** 2) / (2 * sigma_0**2))
    norm = sp.integrate(sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))
    wavefunction /= sp.sqrt(norm)
    uncertainty_x = sp.integrate(x**2 * sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))
    # sigma_0 = sqrt(2 * uncertainty_x) as we use above...
    assert sp.simplify(uncertainty_x - (sigma_0**2 / 2)) == 0


def assert_periodic_width() -> None:
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True)
    sigma_0 = sp.Symbol("sigma_0", real=True, positive=True)
    wavefunction = sp.exp(-(sp.Abs(x) ** 2) / (2 * sigma_0**2))
    norm = sp.integrate(sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))
    wavefunction /= sp.sqrt(norm)
    periodic_x = sp.integrate(
        sp.exp(sp.I * k * x) * sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo)
    )
    sp.print_latex(sp.simplify(periodic_x))

    x_squared = sp.integrate(x**2 * sp.Abs(wavefunction) ** 2, (x, -sp.oo, sp.oo))
    sp.print_latex(sp.simplify(x_squared))


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

    fig, ax = util.get_figure()
    (line,) = ax.plot(temperatures, widths_sse_formula)
    line.set_label("SSE Formula")
    (line,) = ax.plot(temperatures, widths_ballistic_formula)
    line.set_label("Ballistic Formula")
    (line,) = ax.plot(temperatures, widths_coherent)
    line.set_label("Coherent Widths")
    # (line,) = ax.plot(temperatures, widths_free)
    # line.set_label("Free Widths")
    ax.set_ylim(0, None)

    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Width / m")
    ax.set_title("Width of the sodium copper system against temperature")
    ax.legend()
    fig.show()

    util.wait_for_close()

    print(f"{3.615 * angstrom:0.2e}")
