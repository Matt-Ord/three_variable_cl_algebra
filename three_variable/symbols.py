from __future__ import annotations

import sympy as sp
from sympy.physics.units import hbar

# Physical constants
noise = sp.Symbol(r"\xi(t)")
lambda_ = sp.Symbol("lambda", real=True, positive=True)
omega = sp.Symbol("omega", real=True, positive=True)
m = sp.Symbol("m", real=True, positive=True)
KBT = sp.Symbol("KBT", real=True, positive=True)

# Phi and alpha for the coherent state
phi = sp.Symbol(r"\phi", real=True)
alpha = sp.Symbol(r"\alpha")

# Definitions for the squeezing operator, and squeezing derivative
r = sp.Symbol("r", real=True)
theta = sp.Symbol(r"\theta", real=True)
mu_from_r = sp.cosh(r)
nu_from_r_theta = sp.exp(sp.I * theta) * sp.sinh(r)
mu = sp.Symbol(r"\mu", real=True)
nu = sp.Symbol(r"\nu", real=False)
beta = sp.Symbol(r"\beta", real=False)
beta_from_mu_nu = mu / nu


# Dimensionless Parameters for the squeezing operator analysis
eta_m = sp.Symbol("eta_m", real=True, positive=True)
eta_lambda = sp.Symbol("eta_lambda", real=True, positive=True)
eta_omega = sp.Symbol("eta_omega", real=True, positive=True)

m_from_eta_m = hbar**2 * eta_m / (2 * KBT)
lambda_from_eta_lambda = KBT / (hbar * eta_lambda)
omega_from_eta_omega = KBT / (hbar * eta_omega)
