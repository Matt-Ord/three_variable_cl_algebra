from __future__ import annotations

import numpy as np

from three_variable import get_beta, get_beta_0, util

if __name__ == "__main__":
    # Plot of beta against eta_m, for fixed eta and lambda_div_omega
    # we find that |beta_plus| is always above 1, corresponding to an
    # un-physical solution.
    fig0, ax0 = util.get_figure()
    fig1, ax1 = util.get_figure()

    eta_m = np.linspace(0, 100, 1000)
    lambda_div_omega = 1 * np.ones_like(eta_m)
    eta = 10 * np.ones_like(eta_m)

    beta_plus = get_beta(eta_m, eta, lambda_div_omega)
    (line,) = ax0.plot(eta_m, np.abs(beta_plus))
    line.set_label(r"$\beta_+$")
    (line,) = ax1.plot(eta_m, np.angle(beta_plus))
    line.set_label(r"$\beta_+$")

    beta_minus = get_beta(eta_m, eta, lambda_div_omega, positive=False)
    (line,) = ax0.plot(eta_m, np.abs(beta_minus))
    line.set_label(r"$\beta_-$")
    (line,) = ax1.plot(eta_m, np.angle(beta_minus))
    line.set_label(r"$\beta_-$")

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

    # Plot of beta_0 against eta_m - this is what we have in the
    # limit \beta_delta -> 0 ie \eta_omega -> 0
    fig2, ax2 = util.get_figure()
    beta_0 = get_beta_0(eta_m)
    ax2.plot(eta_m, beta_0)

    ax2.set_title(r"Plot of $\beta_0$ against $\eta_m$")
    ax2.set_xlabel(r"$\eta_m$")
    ax2.set_ylabel(r"$\beta_0$")
    fig2.show()

    util.wait_for_close()
