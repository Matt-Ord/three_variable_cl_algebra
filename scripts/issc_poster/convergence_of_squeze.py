from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.util import (
    EtaParameters,
    spaced_time_basis,
)
from scipy.constants import hbar  # type: ignore library
from slate_core import plot
from slate_quantum import operator

from three_variable.physical_systems import TOWNSEND_H_RU
from three_variable.simulation import get_condition_from_params

mpl.rcParams["axes.labelsize"] = 14
if __name__ == "__main__":
    eta_omega, eta_lambda = (
        TOWNSEND_H_RU.eta_parameters.eta_omega,
        TOWNSEND_H_RU.eta_parameters.eta_lambda,
    )
    condition = get_condition_from_params(eta_omega, eta_lambda, mass=1)
    times = spaced_time_basis(n=200, dt=0.002 * np.pi * hbar)
    states = run_stochastic_simulation(condition, times)

    # Similarly the uncertainty
    params = EtaParameters.from_condition(condition)
    widths = operator.measure.all_uncertainty(states, axis=0)
    fig, ax, line = plot.array_against_basis(widths, measure="real")
    ax.set_xlabel("Time /s")
    ax.set_ylabel(r"Uncertainty (/$\hbar$)")
    line.set_linewidth(2)
    line.set_color((0 / 255, 62 / 255, 114 / 255))

    fig.set_size_inches(8, 5)
    fig.tight_layout()
    fig.show()
    fig.savefig(
        f"{Path(__file__).parent}/out/typical_squeeze_dynamics.png",
        dpi=600,  # type: ignore sp
        facecolor="none",  # type: ignore sp
        transparent=True,
    )

    params = EtaParameters.from_condition(condition)
    widths = operator.measure.all_coherent_width(states, axis=0)
    fig, ax, line = plot.array_against_basis(widths, measure="real")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Coherent Width (/m)")
    line.set_linewidth(2)
    line.set_color((0 / 255, 62 / 255, 114 / 255))

    fig.set_size_inches(8, 5)
    fig.tight_layout()
    fig.show()
    fig.savefig(
        f"{Path(__file__).parent}/out/typical_width_dynamics.png",
        dpi=600,  # type: ignore sp
        facecolor="none",  # type: ignore sp
        transparent=True,
    )

    plot.wait_for_close()
