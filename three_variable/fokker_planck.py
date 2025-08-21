from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import sympy as sp
from slate_core.util import timed

from three_variable.simulation import (
    ELENA_LI_CU,
    ELENA_NA_CU,
    TOWNSEND_H_RU,
)
from three_variable.symbols import (
    KBT,
)
from three_variable.util import file_cached

"""
this file computes the Fokker-Planck equation and solves it using a Gaussian ansatz
for a system of stochastic differential equations (SDEs)
"""

# Define x-like and p-like coordinates
t, x, p = sp.symbols("t x p", real=True)
f = sp.Function("f")(x, p)  # Probability density function

# Drift terms and general form of stochastic differential equations
inertia, friction, stiffness = sp.symbols("inertia friction stiffness")
X_force = sp.Symbol("X_force")
drift = [inertia * p + X_force * x, -stiffness * x - friction * p]

# Diffusion matrix B (non-diagonal in general)
b11, b12, b21, b22 = sp.symbols("b11 b12 b21 b22")
B = sp.Matrix([[b11, b12], [b21, b22]])


PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]

# Gaussian ansatz, define the covariance matrix S
z = sp.Matrix([x, p])
s11, s12, s22 = sp.symbols("s11 s12 s22", real=True)
sigma = sp.Matrix([[s11, s12], [s12, s22]])


def derive_fokker_planck(
    variables: list[sp.Symbol], drift_vector: list[sp.Expr], diffusion_matrix: sp.Matrix
) -> sp.Expr:
    """
    Compute the Fokker-Planck equation for a system of SDEs.

    Parameters
    ----------
        variables: list of sympy symbols [x1, x2, ...]
        drift_vector: list of sympy expressions for a_i(x)
        diffusion_matrix: sympy.Matrix of B_ij(x)

    Returns
    -------
        RHS of Fokker-Planck equation (f is probability density function)
    """
    n = len(variables)
    # Drift term: - d/dx_i (a_i f)
    drift_term = sum(-sp.diff(drift_vector[i] * f, variables[i]) for i in range(n))  # type: ignore unknown

    # Diffusion tensor: D_ij = sum_k B_ik B_jk / 2
    D = diffusion_matrix * diffusion_matrix.T / 2

    # Diffusion term: + d²/dx_idx_j (D_ij f) =  + D_ij d²/dx_idx_j (f) since D is constant
    diffusion_term = 0
    for i in range(n):
        for j in range(n):
            diffusion_term += sp.diff(sp.diff(D[i, j] * f, variables[i]), variables[j])  # type: ignore unknown

    # df/dt = drift_term + diffusion_term
    return drift_term + diffusion_term  # type: ignore unknown


def _gaussian_ansatz(fp_eq: sp.Expr, diffusion: Literal["full", "diagonal"]) -> Path:
    return Path(f".cache/gaussian_ansatz.{diffusion}")


@file_cached(_gaussian_ansatz)
@timed
def gaussian_ansatz(
    fp_eq: sp.Expr, diffusion: Literal["full", "diagonal"]
) -> tuple[sp.Expr, ...]:
    """
    Find the steady state solution for Fokker-Planck equation using a Gaussian ansatz.

    Parameters
    ----------
        fp_eq: Fokker-Planck steady state equation.
    """
    # use a diagonal diffusion matrix if requested
    if diffusion == "diagonal":
        fp_eq = fp_eq.subs({b12: 0, b21: 0})

    # define Gaussian ansatz P(x, p) for steady-state distribution
    P = sp.exp(-0.5 * (z.T * sigma * z)[0])  # type: ignore unknown

    # Compute derivatives of P to substitute into fp_eq
    Px = sp.diff(P, x)
    Pp = sp.diff(P, p)
    Pxx = sp.diff(Px, x)
    Ppp = sp.diff(Pp, p)
    Pxp = sp.diff(Px, p)

    # Substitute into FP equation
    fp_subs = (
        fp_eq.subs(
            {
                f: P,
                sp.Derivative(f, x): Px,
                sp.Derivative(f, p): Pp,
                sp.Derivative(f, x, x): Pxx,
                sp.Derivative(f, p, p): Ppp,
                sp.Derivative(f, x, p): Pxp,
            }
        )
        / P
    )
    # Extract coefficient of x, p, xp, x^2, p^2
    expr = sp.simplify(fp_subs).expand()

    coeffs = {
        "x^2": expr.coeff(x, 2),
        "xp": expr.coeff(x).coeff(p),
        "p^2": expr.coeff(p, 2),
        "x": expr.coeff(x) - 2 * expr.coeff(x, 2) * x - expr.coeff(x).coeff(p) * p,
        "p": expr.coeff(p) - 2 * expr.coeff(p, 2) * p - expr.coeff(p).coeff(x) * x,
        "const": expr.subs({x: 0, p: 0}),
    }

    # print coefficients
    print()
    for key, value in coeffs.items():
        sp.print_latex(sp.Eq(sp.Symbol(key), value))
    print()
    # Solve for covariance matrix entries by setting all coeffs to 0
    print("solving")
    solution = sp.solve(list(coeffs.values()), [s11, s12, s22], dict=True)

    # check solution satisfies the equations by substituting back
    assert (
        sp.simplify(
            expr.subs(
                {
                    s11: solution[0][s11],
                    s12: solution[0][s12],
                    s22: solution[0][s22],
                }
            )
        )
        == 0
    ), "Solution does not satisfy Fokker-Planck equation"
    return solution


if __name__ == "__main__":
    # Compute Fokker-Planck
    C = sp.Symbol("C")  # constant
    fp_eq = derive_fokker_planck([x, p], drift, B)
    fp_eq_diagonalized = fp_eq.subs(
        {b21: 0, b12: 0}
    )  # diagonalize the diffusion matrix

    # solve FP equation with Gaussian ansatz centered at zero
    gaussian_solutions = gaussian_ansatz(fp_eq_diagonalized, diffusion="diagonal")
    s11_expr = gaussian_solutions[0][s11]
    s12_expr = gaussian_solutions[0][s12]
    s22_expr = gaussian_solutions[0][s22]
    sigma_ss = sp.Matrix([[s11_expr, s12_expr], [s12_expr, s22_expr]])
    normalization = 2 * np.pi / sp.sqrt(sp.det(sigma_ss))
    f_ss = sp.exp(-0.5 * (z.T * sigma_ss * z)[0]) / normalization  # type: ignore unknown
    sp.print_latex(f_ss)
    # check for Langevin equation
    f_ss_langevin = f_ss.subs(
        {
            X_force: 0,
            b11: 0,
            b22**2: 2 * friction * KBT / inertia,
        }
    )
    sp.print_latex(f_ss_langevin)
