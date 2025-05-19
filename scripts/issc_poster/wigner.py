from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from slate.plot import get_figure, wait_for_close

mpl.rcParams["axes.labelsize"] = 12


def wigner_function(
    q: np.ndarray[Any, np.dtype[np.float64]],
    p: np.ndarray[Any, np.dtype[np.float64]],
    alpha: complex,
    r: float,
    theta: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute the Wigner function of a squeezed coherent state."""
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)

    a = 0.5 * (np.cosh(2 * r) + np.sinh(2 * r) * np.cos(2 * theta))
    b = 0.5 * (np.cosh(2 * r) - np.sinh(2 * r) * np.cos(2 * theta))
    c = -0.5 * np.sinh(2 * r) * np.sin(2 * theta)

    return (1 / np.pi) * np.exp(
        -a * (q - q0) ** 2 - b * (p - p0) ** 2 - c * (q - q0) * (p - p0)
    )


# Define phase space grid
q_vals = np.linspace(-3, 3, 1000)
p_vals = np.linspace(-3, 3, 1000)
Q, P = np.meshgrid(q_vals, p_vals)

# Parameters
alpha = 1.0 + 1.0j  # Coherent displacement
r = 1.0  # Squeezing strength
theta = np.pi / 4  # Squeezing angle

# Compute the Wigner function
W = wigner_function(Q, P, alpha, r, theta)

# Plot
fig, ax = get_figure()
mesh = ax.pcolormesh(Q, P, W, shading="auto", cmap="viridis")
fig.colorbar(mesh)
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$p$")

# Axis lines at (0,0)
line = ax.axhline(0)
line.set_color("C7")
line.set_linestyle("--")
line.set_linewidth(2)
line = ax.axvline(0)
line.set_color("C7")
line.set_linestyle("--")
line.set_linewidth(2)
fig.tight_layout()

fig.set_size_inches(6, 4.5)
fig.set_size_inches(8, 6)
fig.show()
fig.savefig(
    f"{Path(__file__).parent}/out/wigner_function.png", dpi=600, facecolor="none"
)
wait_for_close()
