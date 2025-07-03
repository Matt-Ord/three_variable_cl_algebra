from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sdeint
import sympy as sp
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    action_from_expr,
    extract_action,
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
    zeta,
)

t = sp.Symbol("t", real=True)

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]


alpha_derivative_deterministic = get_full_derivative("alpha")


alpha_derivative_diffusion = get_diffusion_term() / noise
alpha_derivative_diffusion = sp.factor(
    extract_action(action_from_expr(alpha_derivative_diffusion), "alpha")
)
alpha_derivative_diffusion = sp.simplify(
    dimensionless_from_full(alpha_derivative_diffusion),
    rational=True,
)

# .subs(
#     {
#         sp.Symbol("V_1"): 0,
#         eta_lambda: ELENA_NA_CU.eta_parameters.eta_lambda,
#         eta_m: ELENA_NA_CU.eta_parameters.eta_m,
#         eta_omega: ELENA_NA_CU.eta_parameters.eta_omega,
#         KBT: 1.59e-21,
#         hbar: 1.0545718e-34,  # Planck's constant in J.s
#     }
# )

expr_system = get_system_derivative("zeta")
expr_environment = get_environment_derivative("zeta")

zeta_derivative = expr_system + expr_environment

# Substitute physical parameters for numerical evaluation
eta_lambda_value = 1
eta_m_value = 1
eta_omega_value = 1
hbar_value = 1  # Planck's constant in J.s
KBT_value = 1  # Boltzmann constant in J

alpha_derivative_deterministic = alpha_derivative_deterministic.subs(
    {
        sp.Symbol("V_1"): 0,
        eta_lambda: eta_lambda_value,
        eta_m: eta_m_value,
        eta_omega: eta_omega_value,
        KBT: KBT_value,
        hbar: hbar_value,  # Planck's constant in J.s
    }
)
alpha_derivative_diffusion = alpha_derivative_diffusion.subs(
    {
        sp.Symbol("V_1"): 0,
        eta_lambda: eta_lambda_value,
        eta_m: eta_m_value,
        eta_omega: eta_omega_value,
        KBT: KBT_value,
        hbar: hbar_value,  # Planck's constant in J.s
    }
)
zeta_derivative = zeta_derivative.subs(
    {
        sp.Symbol("V_1"): 0,
        eta_lambda: eta_lambda_value,
        eta_m: eta_m_value,
        eta_omega: eta_omega_value,
        KBT: KBT_value,
        hbar: hbar_value,  # Planck's constant in J.s
    }
)


# print the results
# print("Alpha derivative (deterministic):")
# sp.print_latex(alpha_derivative_deterministic)
# print("Alpha derivative (diffusion):")
# sp.print_latex(alpha_derivative_diffusion)
# print("Zeta derivative:")
# sp.print_latex(zeta_derivative)

# input()


# dX = F dt + G dW
# Lambdify to get NumPy-compatible functions
drift_expr = sp.Matrix([alpha_derivative_deterministic, zeta_derivative])
diff_expr = sp.Matrix([[alpha_derivative_diffusion, 0], [0, 0]])

# lambdify full vector input y = [alpha, zeta]
drift_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), drift_expr, modules="numpy")
diff_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), diff_expr, modules="numpy")


# 5. Wrap for sdeint
def f(y, t):
    return np.array(drift_func(t, y)).astype(np.complex128).flatten()


def G(y, t):
    return np.array(diff_func(t, y)).astype(np.complex128)


# 6. Initial values and time vector
y0 = np.array([1.0 + 0.0j, 2.0 + 0.0j])  # alpha and zeta
ts = np.linspace(0, 10, 6000)

print("Starting simulation")
# 7. Solve using Itô interpretation
sol = sdeint.itoint(f, G, y0, ts)

# 8. Extract results
alpha_sol = sol[:, 0]
zeta_sol = sol[:, 1]

print("Simulation completed")
# print("Final alpha:", alpha_sol[-1])
print("Final zeta:", zeta_sol[-1])
# 9. Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ts, alpha_sol.real, label="Re(α)")
plt.plot(ts, alpha_sol.imag, label="Im(α)")
plt.title("Alpha Evolution")
plt.xlabel("Time")
plt.ylabel("Alpha")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(ts, zeta_sol.real, label="Re(ζ)")
plt.plot(ts, zeta_sol.imag, label="Im(ζ)")
plt.title("Zeta Evolution")
plt.xlabel("Time")
plt.ylabel("Zeta")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("alpha_zeta_evolution.png", dpi=300)
print("Plot saved as alpha_zeta_evolution.png")
