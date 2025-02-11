from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann, angstrom, atomic_mass, electron_volt, hbar


@dataclass
class EtaParameters:
    eta_m: float
    eta_lambda: float
    eta_omega: float


@dataclass(frozen=True, kw_only=True)
class BaseParameters:
    m: float
    lambda_: float
    omega: float
    temperature: float

    @property
    def eta_parameters(self) -> EtaParameters:
        eta_m = Boltzmann * self.temperature / (hbar**2 / (2 * self.m))
        eta_lambda = Boltzmann * self.temperature / (self.lambda_ * hbar)
        eta_omega = Boltzmann * self.temperature / (hbar * self.omega)
        return EtaParameters(eta_m=eta_m, eta_lambda=eta_lambda, eta_omega=eta_omega)

    def with_temperature(self, temperature: float) -> BaseParameters:
        return BaseParameters(
            m=self.m, lambda_=self.lambda_, omega=self.omega, temperature=temperature
        )


@dataclass
class ElenaParameters:
    m: float
    lambda_: float
    barrier_energy: float
    lattice_parameter: float
    temperature: float

    @property
    def base_parameters(self) -> BaseParameters:
        # V(x) = (E_b / 2) * cos(k x) where k = 4 pi / (sqrt(3) a)
        #      = (E_b / 2) * (1 - (k x)**2 / 2 + ... )
        #      = A - (E_b / 4) * (k)**2 x**2
        k = (4 * np.pi) / (np.sqrt(3) * self.lattice_parameter)
        v_2 = (self.barrier_energy / 4) * (k) ** 2
        #      = A - 0.5 * m * omega**2 * x**2
        # so omega is = sqrt(2 * v_2 / m)
        omega = np.sqrt(2 * v_2 / self.m)
        return BaseParameters(
            m=self.m, lambda_=self.lambda_, omega=omega, temperature=self.temperature
        )

    @property
    def eta_parameters(self) -> EtaParameters:
        return self.base_parameters.eta_parameters


NA_MASS = 22.990 * atomic_mass
LI_MASS = 6.94 * atomic_mass
HYDROGEN_MASS = 1.008 * atomic_mass

ELENA_CU_LATTICE_PARAMETER = 3.615 * angstrom

ELENA_NA_CU = ElenaParameters(
    m=NA_MASS,
    barrier_energy=55 * 10**-3 * electron_volt,
    lattice_parameter=ELENA_CU_LATTICE_PARAMETER,
    temperature=155,
    lambda_=0.2 * 10**12,
)

ELENA_NA_CU_BALLISTIC = ElenaParameters(
    m=NA_MASS,
    barrier_energy=55 * 10**-3 * electron_volt,
    lattice_parameter=ELENA_CU_LATTICE_PARAMETER / np.sqrt(3),
    temperature=155,
    lambda_=0.2 * 10**12,
)

ELENA_LI_CU = ElenaParameters(
    m=LI_MASS,
    barrier_energy=45 * 10**-3 * electron_volt,
    lattice_parameter=ELENA_CU_LATTICE_PARAMETER,
    temperature=140,
    lambda_=1.2 * 10**12,
)

TOWNSEND_H_RU = BaseParameters(
    m=HYDROGEN_MASS,
    lambda_=4 * 10**12,
    omega=113 * 10**12 / (2 * np.pi),
    temperature=170,
)
