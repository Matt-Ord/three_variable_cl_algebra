from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.util import spaced_time_basis
from scipy.constants import hbar  # type: ignore libary
from slate import array, plot
from slate_quantum import dynamics

from three_variable.simulation import get_condition_from_params

mpl.rcParams["axes.labelsize"] = 12
eta_omega, eta_lambda = 0.5, 40.0
condition = get_condition_from_params(eta_omega, eta_lambda, mass=1)
# We simulate the system using the stochastic Schrodinger equation.
# We find a localized stochastic evolution of the wavepacket.
times = spaced_time_basis(n=100, dt=0.1 * np.pi * hbar)
states = run_stochastic_simulation(condition, times)
states = dynamics.select_realization(states)

# We start the system in a gaussian state, centered at the origin.
fig, ax, line1 = plot.array_against_axes_1d(states[50, :], measure="abs")
line1.set_linewidth(2)
line1.set_color((0 / 255, 62 / 255, 114 / 255))
line1.set_label("Typical State")

ax1 = ax.twinx()
_, _, line0 = plot.array_against_axes_1d(
    array.as_outer_array(condition.potential), ax=ax1
)
line0.set_linewidth(2)
line0.set_color("black")
line0.set_label("Potential")


ax.set_xlim(0, states.basis.metadata()[1][0].delta)
ax.set_ylabel("Occupation")
ax.set_xlabel("Position (/m)")
ax1.set_ylabel("Potential (/J)")
ax.legend(handles=[line1, line0])
fig.set_size_inches(6, 6)
fig.tight_layout()
fig.show()
fig.savefig(
    f"{Path(__file__).parent}/out/typical_state.png",
    dpi=600,
    facecolor="none",
    transparent=True,
)

plot.wait_for_close()
