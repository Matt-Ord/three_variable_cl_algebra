from __future__ import annotations

import numpy as np

from three_variable import get_beta, util

if __name__ == "__main__":
    # Plot of beta against m, for a range of eta_lambda
    # we find that |beta_plus| is always above 1, so this is ignored
    # we find that |beta_minus| has a lower minimum value for larger eta
    # and the m value at which this minimum occurs increases with eta
    # The phase of beta_minus moves from 0 to -pi as m increases

    eta_m = np.linspace(0, 100, 1000)
    for eta_omega in [0.3, 0.6]:
        fig0, ax0 = util.get_figure()
        fig1, ax1 = util.get_figure()

        for eta_lambda in [1, 5, 10, 50, 100, 500, 1000]:
            eta_lambda_array = eta_lambda * np.ones_like(eta_m)
            eta_omega_array = eta_omega * np.ones_like(eta_m)

            beta_minus = get_beta(
                eta_m, eta_omega_array, eta_lambda_array, positive=True
            )
            (line,) = ax0.plot(eta_m, np.abs(beta_minus))
            line.set_label(rf"$\eta_\lambda = {eta_lambda}$")
            (line,) = ax1.plot(eta_m, np.angle(beta_minus))
            line.set_label(rf"$\eta_\lambda = {eta_lambda}$")

        ax0.set_title(
            rf"Plot of $|{{\beta}}|$ against $\eta_m$ for $\eta_\omega = {eta_omega}$"
        )
        ax0.set_xlabel(r"$\eta_m$")
        ax0.set_ylabel(r"$|{\beta}|$")
        ax0.legend()
        fig0.show()
        ax1.set_title(
            rf"Plot of arg{{$\beta$}} against $\eta_m$ for $\eta_\omega = {eta_omega}$"
        )
        ax1.set_xlabel(r"$\eta_m$")
        ax1.set_ylabel(r"arg{$\beta$}")
        ax1.legend()
        fig1.show()
    util.wait_for_close()
