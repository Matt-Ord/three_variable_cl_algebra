from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp
from classical_equation_of_motion import (
    get_classical_equilibrium_derivative,
)
from plot_coefficients import plot_2d_real_imag_heatmaps
from sympy.physics.units import hbar

from three_variable.equilibrium_squeeze import (
    get_equilibrium_squeeze_ratio,
    get_equilibrium_zeta,
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.simulation import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU
from three_variable.symbols import (
    KBT,
    eta_lambda,
    eta_m,
    eta_omega,
    noise,
    p,
    x,
)

Re_xi = sp.Symbol("Re_xi", real=True)
Im_xi = sp.Symbol("Im_xi", real=True)


def get_numerical_derivative2() -> tuple[sp.Expr, sp.Expr]:
    x_derivative = get_classical_equilibrium_derivative("x").subs(
        {
            sp.Symbol("V_1"): 0,
            noise: Re_xi + 1j * Im_xi,
            sp.conjugate(noise): Re_xi - 1j * Im_xi,
        }
    )
    p_derivative = get_classical_equilibrium_derivative("p").subs(
        {
            sp.Symbol("V_1"): 0,
            noise: Re_xi + 1j * Im_xi,
            sp.conjugate(noise): Re_xi - 1j * Im_xi,
        }
    )
    x_derivative = squeeze_ratio_from_zeta_expr(x_derivative)
    p_derivative = squeeze_ratio_from_zeta_expr(p_derivative)
    return (x_derivative, p_derivative)


def evaluate_equilibrium_squeeze(
    eta_m_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_omega_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_lambda_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    hbar_value: float,
    KBT_value: float,
    positive: bool = True,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    expression = get_equilibrium_zeta(positive=positive)
    expression = expression.subs({KBT: KBT_value, hbar: hbar_value})
    expression_lambda = sp.lambdify(
        (eta_m, eta_omega, eta_lambda), expression, modules="numpy"
    )
    return expression_lambda(eta_m_values, eta_omega_values, eta_lambda_values)  # type: ignore[no-untyped-call]


def evaluate_equilibrium_ratio(
    eta_m_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_omega_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_lambda_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    hbar_value: float,
    KBT_value: float,
    positive: bool = True,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    expression = get_equilibrium_squeeze_ratio(positive=positive)
    expression = expression.subs({KBT: KBT_value, hbar: hbar_value})
    expression_lambda = sp.lambdify(
        (eta_m, eta_omega, eta_lambda), expression, modules="numpy"
    )
    return expression_lambda(eta_m_values, eta_omega_values, eta_lambda_values)  # type: ignore[no-untyped-call]


def lambdify_to_eta(
    expr: sp.Expr,
    eta_m_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_omega_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    eta_lambda_values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    KBT_value: float,
    hbar_value: float,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """Substitute constants and evaluate lambdified expressions as a function of eta params and zeta."""
    substituted_expr = expr.subs(
        {
            KBT: KBT_value,
            hbar: hbar_value,
        }
    )

    lambdified_expr = sp.lambdify(
        (eta_m, eta_omega, eta_lambda, squeeze_ratio), substituted_expr, modules="numpy"
    )

    ratio_values = evaluate_equilibrium_ratio(
        eta_m_values, eta_omega_values, eta_lambda_values, hbar_value, KBT_value
    )
    return lambdified_expr(
        eta_m_values, eta_omega_values, eta_lambda_values, ratio_values
    )


if __name__ == "__main__":
    # Example usage
    PHYSICAL_PARAMS = [
        ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
        ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
        ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
    ]
    eta_params = PHYSICAL_PARAMS[2][1]
    # convert to normalized units
    normalized_eta_params = eta_params.to_normalized()
    eta_m_values = np.array([normalized_eta_params.eta_m])
    KBT_value = 1
    hbar_value = 1
    eta_lambda_values = np.logspace(-20, 20, num=500)
    eta_omega_values = np.logspace(-10, 20, num=500)
    # meshgrid for eta parameters
    eta_m_values, eta_omega_values, eta_lambda_values = np.meshgrid(
        eta_m_values, eta_omega_values, eta_lambda_values, indexing="ij"
    )
    print("getting analytical derivatives")
    x_derivative, p_derivative = get_numerical_derivative2()
    print()
    print("x derivative:")
    sp.print_latex(x_derivative)
    print("p derivative:")
    sp.print_latex(p_derivative)
    input()
    # get the coefficients in the x and p derivatives
    coeffs = {
        "dxdt_coeff_x": x_derivative.subs({x: 1, p: 0, Re_xi: 0, Im_xi: 0})
        * eta_params.kbt_div_hbar,
        "dxdt_coeff_p": x_derivative.subs({x: 0, p: 1, Re_xi: 0, Im_xi: 0}),
        "dpdt_coeff_x": p_derivative.subs({x: 1, p: 0, Re_xi: 0, Im_xi: 0}),
        "dpdt_coeff_p": p_derivative.subs({x: 0, p: 1, Re_xi: 0, Im_xi: 0}),
        "dxdt_coeff_Re_xi": x_derivative.subs({x: 0, p: 0, Re_xi: 1, Im_xi: 0}),
        "dxdt_coeff_Im_xi": x_derivative.subs({x: 0, p: 0, Re_xi: 0, Im_xi: 1}),
        "dpdt_coeff_Re_xi": p_derivative.subs({x: 0, p: 0, Re_xi: 1, Im_xi: 0}),
        "dpdt_coeff_Im_xi": p_derivative.subs({x: 0, p: 0, Re_xi: 0, Im_xi: 1}),
    }
    # Evaluate the coefficients for each physical parameter
    results = {}
    print()
    print("evaluating coefficients")
    # loop through the coeffs dictionary and evaluate each coefficient
    for key, value in coeffs.items():
        results[key] = lambdify_to_eta(
            value,
            eta_m_values=eta_m_values,
            eta_omega_values=eta_omega_values,
            eta_lambda_values=eta_lambda_values,
            KBT_value=KBT_value,  # Example temperature in Kelvin
            hbar_value=hbar_value,
        )
    # flatten the results to convert to dataframe
    print()
    print("converting to dataframe")
    for key, value in results.items():
        results[key] = value.flatten()
    coeffs_df = pd.DataFrame(results)
    coeffs_df["eta_m"] = eta_m_values.flatten()
    coeffs_df["eta_omega"] = eta_omega_values.flatten()
    coeffs_df["eta_lambda"] = eta_lambda_values.flatten()
    coeffs_df["zeta"] = evaluate_equilibrium_squeeze(
        eta_m_values, eta_omega_values, eta_lambda_values, hbar_value, KBT_value
    ).flatten()
    coeffs_df["ratio"] = evaluate_equilibrium_ratio(
        eta_m_values, eta_omega_values, eta_lambda_values, hbar_value, KBT_value
    ).flatten()
    coeffs_df["lambda_eff_factor"] = (
        -coeffs_df["dpdt_coeff_p"] * coeffs_df["eta_lambda"] * hbar_value / KBT_value
    )

    coeffs_df["mass_eff_factor"] = (
        2 * KBT_value / (hbar_value**2 * coeffs_df["eta_m"] * coeffs_df["dxdt_coeff_p"])
    )

    coeffs_df["omega_eff_factor"] = (
        np.sqrt(
            -2
            * coeffs_df["dpdt_coeff_x"]
            / (coeffs_df["eta_m"] * KBT_value * coeffs_df["mass_eff_factor"])
        )
        * coeffs_df["eta_omega"]
    )
    # print()
    # print("saving coefficients to csv")
    # coeffs_df.to_csv("coefficients.csv", index=False)
    print("plotting coefficients")
    params_to_plot = [
        "dxdt_coeff_x",
        "mass_eff_factor",
        "omega_eff_factor",
        "lambda_eff_factor",
        "ratio",
    ]
    for _param in params_to_plot:
        plot_2d_real_imag_heatmaps(
            coeffs=coeffs_df,
            param_x="eta_lambda",
            param_y="eta_omega",
            value_col=_param,
        )
