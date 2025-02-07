from __future__ import annotations

import numpy as np

from three_variable import get_beta, util

if __name__ == "__main__":
    # Plot of beta against eta_m, for a range of eta_omega
    # we find that |beta_plus| is always above 1, so this is ignored
    # we find that |beta_minus| becomes positive for small eta,
    # corresponding to a large harmonic oscillator frequency
    # (ie much larger than the frequency of the friction)
    # This is because beta_1 becomes greater than 1.
    fig0, ax0 = util.get_figure()
    fig1, ax1 = util.get_figure()

    eta_m = np.linspace(0, 100, 1000)
    eta_lambda = 1 * np.ones_like(eta_m)

    delta = 0.000001
    for eta_omega in [0.1, 0.2, 0.3, 0.4, 0.5 - delta, 0.5 + delta, 0.6, 0.7, 0.8]:
        eta_omega_array = eta_omega * np.ones_like(eta_m)

        beta_minus = get_beta(eta_m, eta_omega_array, eta_lambda, positive=False)
        (line,) = ax0.plot(eta_m, np.abs(beta_minus))
        line.set_label(rf"$\eta = {eta_omega}$")
        (line,) = ax1.plot(eta_m, np.angle(beta_minus))
        line.set_label(rf"$\eta = {eta_omega}$")

    ax0.set_title(r"Plot of $|{\beta}|$ against $\eta_m$")
    ax0.set_xlabel(r"$\eta_m$")
    ax0.set_ylabel(r"$|{\beta}|$")
    ax0.legend()
    fig0.show()
    ax1.set_title(r"Plot of arg{$\beta$}} against $\eta_m$")
    ax1.set_xlabel(r"$\eta_m$")
    ax1.set_ylabel(r"arg{$\beta$}")
    ax1.legend()
    fig1.show()
    util.wait_for_close()
