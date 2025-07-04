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


def get_numerical_derivatives(
    alpha_derivative: sp.Expr,
    zeta_derivative: sp.Expr,
    eta_lambda_value: float,
    eta_m_value: float,
    eta_omega_value: float,
    KBT_value: float,
    hbar_value: float,
) -> tuple[sp.Expr, sp.Expr]:
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

    # plug in equilibrium zeta into alpha derivative and diffusion
    alpha_derivative_numerical = sp.expand(alpha_derivative.subs(zeta, zeta_eq))
    alpha_derivative_numerical = sp.collect(
        alpha_derivative_numerical.evalf(), [alpha, sp.conjugate(alpha)]
    )
    return zeta_eq, alpha_derivative_numerical


t = sp.Symbol("t", real=True)

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]

# print("Physical parameters:")
# for name, params, color in PHYSICAL_PARAMS:
#     print(f"{name}: {params}")
#     print(f"Color: {color}")
# input()

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


# # print the symbolic expressions
# print("Alpha derivative (deterministic):")
# sp.print_latex(alpha_derivative_deterministic)
# # sp.print_latex(sp.limit(alpha_derivative_deterministic, zeta, -1))
# print("Alpha derivative (diffusion):")
# sp.print_latex(alpha_derivative_diffusion * noise)
# print("Zeta derivative:")
# sp.print_latex(zeta_derivative)

# input()

# Substitute physical parameters for numerical evaluation
eta_lambda_value = 0.01
eta_m_value = 1e5
eta_omega_value = 1
hbar_value = 1
KBT_value = 1

# eta_lambda_value = ELENA_NA_CU.eta_parameters.eta_lambda
# eta_m_value = ELENA_NA_CU.eta_parameters.eta_m
# eta_omega_value = ELENA_NA_CU.eta_parameters.eta_omega
# KBT_value = 1.59e-21
# hbar_value = 1.0545718e-34

# get numerical derivatives
zeta_eq_numerical, alpha_derivative_numerical = get_numerical_derivatives(
    alpha_derivative,
    zeta_derivative,
    eta_lambda_value,
    eta_m_value,
    eta_omega_value,
    KBT_value,
    hbar_value,
)

print("Equilibrium zeta:", zeta_eq_numerical)
print("Alpha derivative at equilibrium zeta:")
sp.print_latex(alpha_derivative_numerical)

input()

# dX = F dt + G dW
# Lambdify to get NumPy-compatible functions
drift_expr = sp.Matrix([alpha_derivative_deterministic, zeta_derivative])
diff_expr = sp.Matrix([[alpha_derivative_diffusion / noise, 0], [0, 0]])

# lambdify full vector input y = [alpha, zeta]
drift_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), drift_expr, modules="numpy")
diff_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), diff_expr, modules="numpy")


# 5. Wrap for sdeint
def f(y, t):
    return np.array(drift_func(t, y)).astype(np.complex128).flatten()


def G(y, t):
    return np.array(diff_func(t, y)).astype(np.complex128)


# 6. Initial values and time vector
y0 = np.array([1.0 + 0.0j, -2 / 3 + 0.0j])  # alpha and zeta
ts = np.linspace(0, 5, int(eta_m_value * 1000))

print("Starting simulation")
# 7. Solve using Itô interpretation
sol = sdeint.itoint(f, G, y0, ts)

# ts = np.linspace(0, 0.005, 1000)
# sol = sdeint.itoint(f, G, sol[-1, :], ts)

# 8. Extract results
alpha_sol = sol[:, 0]
zeta_sol = sol[:, 1]

# 9. calculate x and p from alpha and zeta
zeta_conj = np.conjugate(zeta_sol)
alpha_conj = np.conjugate(alpha_sol)
a_expec = (alpha_sol + alpha_conj * zeta_sol) / (1 - zeta_sol * zeta_conj)
a_dagger_expec = (alpha_conj + alpha_sol * zeta_conj) / (1 - zeta_sol * zeta_conj)
x_sol = (a_expec + a_dagger_expec) / np.sqrt(2)
p_sol = 1j * (-a_expec + a_dagger_expec) / np.sqrt(2)
dxdt = np.gradient(x_sol, ts)
dpdt = np.gradient(p_sol, ts)

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

# plot the last 100 points x and p
plt.figure(figsize=(12, 6))

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
(line,) = ax1.plot(ts[-100:], (dxdt[-100:].real / 2) / eta_m_value, label="Re(dxdt)")
line.set_marker("x")
ax2.plot(ts[-100:], dpdt[-100:].real, label="Re(dpdt)")
# ax1.title("X Evolution (Last 100 Points)")
# ax1.xlabel("Time")
# ax1.ylabel("X")
# ax1.legend()

ax1.plot(ts[-100:], p_sol[-100:].real, label="Re(p)")
ax2.plot(ts[-100:], dpdt[-100:].imag, label="Im(dpdt)")
# plt.title("P Evolution (Last 100 Points)")
# plt.xlabel("Time")
# plt.ylabel("P")
# plt.legend()
# plt.tight_layout()
# plt.grid()
plt.savefig("x_p_evolution_last_100.png", dpi=300)
print("Plot saved as x_p_evolution_last_100.png")
