from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.constants import (  # type: ignore scipy
    Boltzmann,
)
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    expectation_from_formula,
)
from three_variable.equilibrium_squeeze import (
    squeeze_ratio,
    squeeze_ratio_from_zeta_expr,
)
from three_variable.projected_sse import get_harmonic_term, get_kinetic_term
from three_variable.simulation import (
    EtaParameters,
    SimulationConfig,
    evaluate_equilibrium_squeeze_ratio,
    hbar_value,
    run_projected_simulation,
)
from three_variable.symbols import (
    KBT,
    alpha,
    dimensionless_from_full,
    eta_lambda,
    eta_m,
    eta_omega,
    formula_from_expr,
)


# run numerical simulation to get alpha(t) and r(t)
# from the simulation, calculate the probability distribution of energy in steady state
def get_energy_from_alpha(
    alpha_value: np.ndarray,
    eta_lambda_val: float,
    eta_omega_val: float,
    eta_m_val: float,
    KBT_val: float,
) -> sp.Expr:
    """Calculate the energy distribution of the system."""
    energy = get_kinetic_term() + get_harmonic_term()
    energy = formula_from_expr(energy)
    energy = expectation_from_formula(energy)
    energy = squeeze_ratio_from_zeta_expr(energy)
    r_eq = evaluate_equilibrium_squeeze_ratio(
        eta_lambda_val=eta_lambda_val,
        eta_omega_val=eta_omega_val,
    )
    energy = dimensionless_from_full(energy)
    energy = energy.subs(
        {
            eta_lambda: eta_lambda_val,
            eta_omega: eta_omega_val,
            eta_m: eta_m_val,
            squeeze_ratio: r_eq,
            hbar: hbar_value,
            KBT: KBT_val,
        }
    )
    energy_fn = sp.lambdify(
        (alpha),
        energy,
        modules="numpy",
    )
    return energy_fn(alpha_value)


if __name__ == "__main__":
    eta_m_val = 1e22
    eta_lambda_val = 60
    eta_omega_val = 1
    KBT_value = Boltzmann * 200

    time_scale = hbar_value / KBT_value
    print("Estimating initial r0")
    r0_initial_estimate = evaluate_equilibrium_squeeze_ratio(
        eta_lambda_val=eta_lambda_val,
        eta_omega_val=eta_omega_val,
    )
    print("Running simulation")
    solution = run_projected_simulation(
        SimulationConfig(
            params=EtaParameters(
                eta_lambda=eta_lambda_val,
                eta_m=eta_m_val,
                eta_omega=eta_omega_val,
                kbt_div_hbar=KBT_value / hbar_value,
            ),
            alpha_0=0.000001 + 0.0j,
            r_0=r0_initial_estimate,
            times=np.linspace(0, 10000, 500000) * time_scale,
        )
    )
    print("Simulation completed")
    print("equilibrium squeeze ratio:", solution.squeeze_ratio[-1])
    print("Calculating energy distribution")
    # get x and p and thus energy of the system
    steady_state = solution[-50000::]
    energy_values = np.real(
        get_energy_from_alpha(
            alpha_value=steady_state.alpha,
            eta_lambda_val=eta_lambda_val,
            eta_omega_val=eta_omega_val,
            eta_m_val=eta_m_val,
            KBT_val=KBT_value,
        )
    )
    # plot a histogram of the energy
    print("Plotting energy distribution")
    energy_hist, energy_bins = np.histogram(energy_values, bins=10)
    energy_bins = (energy_bins[:-1] + energy_bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.plot(energy_bins, np.log(energy_hist), label="Energy Distribution")
    print(np.log(energy_hist)[0])
    ax.set_xlabel("Energy")
    ax.set_ylabel("Count")
    ax.set_title("Energy Distribution of the System")
    # plot a straight line with slope 1/KBT_value
    kbt_line = -(energy_bins - energy_bins[0]) / KBT_value + np.log(energy_hist)[0]
    ax.plot(energy_bins, kbt_line, label="1/KBT Line", linestyle="--", color="orange")
    ax.legend()
    fig.savefig("energy_distribution_simulation.png", dpi=300)
    fig.clear()
    plt.close(fig)
