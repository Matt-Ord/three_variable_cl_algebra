from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    action_from_expr,
    extract_action,
    xp_expression_from_alpha,
)
from three_variable.projected_sse import (
    get_diffusion_term,
    get_environment_derivative,
    get_full_derivative,
    get_system_derivative,
)
from three_variable.simulation import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU
from three_variable.symbols import (
    KBT,
    alpha,
    dimensionless_from_full,
    eta_lambda,
    eta_m,
    eta_omega,
    noise,
    p,
    x,
    zeta,
)

Re_xi = sp.Symbol("Re_xi")
Im_xi = sp.Symbol("Im_xi")


def get_numerical_derivatives(
    alpha_derivative: sp.Expr,
    zeta_derivative: sp.Expr,
    eta_lambda_value: float,
    eta_m_value: float,
    eta_omega_value: float,
    KBT_value: float,
    hbar_value: float,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Get numerical time derivatives for the alpha, x, and p at equilibrium zeta."""
    alpha_derivative = alpha_derivative.subs(
        {
            sp.Symbol("V_1"): 0,
            eta_lambda: eta_lambda_value,
            eta_m: eta_m_value,
            eta_omega: eta_omega_value,
            KBT: KBT_value,
            hbar: hbar_value,
        }
    )
    zeta_derivative = zeta_derivative.subs(
        {
            sp.Symbol("V_1"): 0,
            eta_lambda: eta_lambda_value,
            eta_m: eta_m_value,
            eta_omega: eta_omega_value,
            KBT: KBT_value,
            hbar: hbar_value,
        }
    )
    # get equilibrium zeta
    zeta_eq = sp.solve(zeta_derivative, zeta)[0]

    # plug in equilibrium zeta into all derivatives and collect terms
    alpha_derivative = sp.expand(alpha_derivative.subs(zeta, zeta_eq))
    alpha_derivative = sp.collect(
        alpha_derivative.evalf(), [alpha, sp.conjugate(alpha)]
    )
    # get x and p derivatives
    alpha_derivative_conj = sp.conjugate(alpha_derivative)
    a_expec_derivative = (alpha_derivative + alpha_derivative_conj * zeta) / (
        1 - zeta * sp.conjugate(zeta)
    )
    a_dagger_expec_derivative = (
        alpha_derivative_conj + alpha_derivative * sp.conjugate(zeta)
    ) / (1 - zeta * sp.conjugate(zeta))
    dxdt = (a_expec_derivative + a_dagger_expec_derivative) / np.sqrt(2)
    dpdt = (
        1j * hbar_value * (-a_expec_derivative + a_dagger_expec_derivative) / np.sqrt(2)
    )
    # substitute numerical values to all derivatives
    dxdt_num = xp_expression_from_alpha(dxdt).subs(
        {
            sp.Symbol("V_1"): 0,
            eta_lambda: eta_lambda_value,
            eta_m: eta_m_value,
            eta_omega: eta_omega_value,
            KBT: KBT_value,
            hbar: hbar_value,
            noise: Re_xi + 1j * Im_xi,
            sp.conjugate(noise): Re_xi - 1j * Im_xi,
        }
    )
    dpdt_num = xp_expression_from_alpha(dpdt).subs(
        {
            sp.Symbol("V_1"): 0,
            eta_lambda: eta_lambda_value,
            eta_m: eta_m_value,
            eta_omega: eta_omega_value,
            KBT: KBT_value,
            hbar: hbar_value,
            noise: Re_xi + 1j * Im_xi,
            sp.conjugate(noise): Re_xi - 1j * Im_xi,
        }
    )

    dxdt_num = sp.expand(dxdt_num.subs(zeta, zeta_eq))
    dpdt_num = sp.expand(dpdt_num.subs(zeta, zeta_eq))
    dxdt_num = sp.collect(dxdt_num.evalf(), [x, p, Re_xi, Im_xi])
    dpdt_num = sp.collect(dpdt_num.evalf(), [x, p, Re_xi, Im_xi])

    return (zeta_eq, alpha_derivative, dxdt_num, dpdt_num)


t = sp.Symbol("t", real=True)

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]


alpha_derivative_deterministic = get_full_derivative("alpha")

alpha_derivative_diffusion = get_diffusion_term()
alpha_derivative_diffusion = sp.factor(
    extract_action(action_from_expr(alpha_derivative_diffusion), "alpha")
)
alpha_derivative_diffusion = sp.simplify(
    dimensionless_from_full(alpha_derivative_diffusion),
    rational=True,
)

alpha_derivative = alpha_derivative_deterministic + alpha_derivative_diffusion

expr_system = get_system_derivative("zeta")
expr_environment = get_environment_derivative("zeta")

zeta_derivative = expr_system + expr_environment


# Substitute physical parameters for numerical evaluation
eta_lambda_value = 1
eta_m_value = 1
eta_omega_value = 1
hbar_value = 1e-3
KBT_value = 1e-3

# eta_lambda_value = ELENA_NA_CU.eta_parameters.eta_lambda
# eta_m_value = ELENA_NA_CU.eta_parameters.eta_m
# eta_omega_value = ELENA_NA_CU.eta_parameters.eta_omega
# KBT_value = 1.59e-21
# hbar_value = 1.0545718e-34


# iterate over values of eta_lambda in log space
eta_lambda_values = np.logspace(6, 6, num=1)
eta_omega_values = np.logspace(1, 1, num=1)
eta_m_values = np.logspace(1, 1, num=1)
results = []

for eta_lambda_value in eta_lambda_values:
    for eta_m_value in eta_m_values:
        for eta_omega_value in eta_omega_values:
            # get numerical derivatives
            (
                zeta_eq,
                alpha_derivative,
                dxdt_num,
                dpdt_num,
            ) = get_numerical_derivatives(
                alpha_derivative,
                zeta_derivative,
                eta_lambda_value,
                eta_m_value,
                eta_omega_value,
                KBT_value,
                hbar_value,
            )
            # Extract coefficients
            result = {
                "eta_lambda": eta_lambda,
                "eta_m": eta_m,
                "eta_omega": eta_omega,
                "zeta_eq": zeta_eq,
                "dxdt_coeff_x": dxdt_num.coeff(x),
                "dxdt_coeff_p": dxdt_num.coeff(p),
                "dxdt_coeff_Re_xi": dxdt_num.coeff(Re_xi),
                "dxdt_coeff_Im_xi": dxdt_num.coeff(Im_xi),
                "dpdt_coeff_x": dpdt_num.coeff(x),
                "dpdt_coeff_p": dpdt_num.coeff(p),
                "dpdt_coeff_Re_xi": dpdt_num.coeff(Re_xi),
                "dpdt_coeff_Im_xi": dpdt_num.coeff(Im_xi),
            }
            results.append(result)

print(results)
