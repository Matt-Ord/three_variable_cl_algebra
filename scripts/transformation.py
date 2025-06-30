from __future__ import annotations

import sympy as sp

from three_variable.symbols import (
    zeta,
)

# define the constants
s = sp.Symbol("s", real=True)  # sin
c = sp.Symbol("c", real=True)  # cos
r = sp.Symbol("r", real=True)  # r = |zeta|
F = sp.Symbol("F", complex=True)  # system derivative of alpha
Re = sp.re(F)
Im = sp.im(F)
E = sp.Symbol("E", real=True)  # E = exp(2*r)
R = sp.Symbol("R", real=True)  # R = (1 - zeta) / (1 + zeta)

# define the transformation matrices
Rotation = sp.Matrix([[c, s], [-s, c]])
Rotation_inv = sp.Matrix([[c, -s], [s, c]])
P = sp.Matrix([[sp.exp(-r), 0], [0, sp.exp(r)]])

K_alpha_to_x = sp.Matrix([[1, 1], [-sp.I, sp.I]]) / sp.sqrt(
    2
)  # K_alpha_to_x is the transformation matrix: x = K_alpha_to_x * alpha

K_alpha_to_a = sp.Matrix(
    [
        [sp.conjugate(zeta), 1],
        [1, zeta],
    ]
) / (
    1 - sp.Abs(zeta) ** 2
)  # K_alpha_to_a is the transformation matrix: a = K_alpha_to_a * alpha

# S_a is the squeezing matrix in the ladder operator basis
S_a = (
    sp.Matrix(
        [
            [E + 1 / E, -(E - 1 / E) * (c + sp.I * s)],
            [-(E - 1 / E) * (c - sp.I * s), E + 1 / E],
        ]
    )
    / 2
)

# S_alpha is the squeezing matrix in the alpha basis
S_alpha = K_alpha_to_a.inv() * S_a * K_alpha_to_a

# S_x is the squeezing matrix in the (x, p) basis
S_x = K_alpha_to_x * S_alpha * K_alpha_to_x.inv()

# Simplify the squeezing matrix S_x
S_x = S_x.applyfunc(lambda expr: expr.subs({c**2: 1 - s**2, zeta: (1 - R) / (1 + R)}))
S_x = sp.simplify(S_x)

sp.print_latex(S_x)
input()

A, B, C, D = sp.symbols("A B C D", real=True)

S_inv = S.inv()
M = sp.Matrix([[Re, -Im], [Im, Re]])
M_full_derivative = sp.Matrix([[A, B], [C, D]])

# transformation = S.inv() * M * S
transformation = S.inv() * M_full_derivative * S
transformation = transformation.applyfunc(
    lambda expr: expr.subs({sp.exp(2 * r): E, c**2: 1 - s**2})
)
transformation = transformation.applyfunc(lambda expr: sp.collect(sp.expand(expr), [E]))

sp.print_latex(transformation)
