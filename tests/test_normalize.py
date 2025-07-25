from __future__ import annotations

import numpy as np

from three_variable.simulation.physical_systems import (
    BaseParameters,
    EtaParameters,
    Units,
)


def test_nomalize_eta() -> None:
    eta = EtaParameters(
        eta_m=20,
        eta_lambda=1.0,
        eta_omega=1.0,
        kbt_div_hbar=100,
        units=Units(),
    )
    normalized_eta = eta.to_normalized()
    assert np.isclose(normalized_eta.eta_m, 1)
    assert normalized_eta.m == eta.m

    assert normalized_eta.kbt_div_hbar == 1
    assert normalized_eta.temperature == eta.temperature
    assert normalized_eta.eta_lambda == 1
    assert normalized_eta.lambda_ == eta.lambda_

    assert normalized_eta.eta_omega == 1
    assert normalized_eta.omega == eta.omega


def test_into_eta() -> None:
    params = BaseParameters(
        m=10,
        lambda_=5,
        omega=5,
        temperature=300,
    )
    assert np.isclose(params.m, params.eta_parameters.m)
    assert np.isclose(params.lambda_, params.eta_parameters.lambda_)
    assert np.isclose(params.omega, params.eta_parameters.omega)
    assert np.isclose(params.temperature, params.eta_parameters.temperature)
