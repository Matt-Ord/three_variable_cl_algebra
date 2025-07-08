from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np
from scipy.constants import (  # type: ignore scipy
    Boltzmann,
    angstrom,
    atomic_mass,
    electron_volt,
    hbar,
)


@dataclass(frozen=True, kw_only=True)
class Units:
    time_factor: float = 1.0
    length_factor: float = 1.0

    @overload
    def length_into(self, value: float, units: Units) -> float: ...
    @overload
    def length_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: Units
    ) -> np.ndarray[Any, DT]: ...
    def length_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: Units
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        return value * (self.length_factor / units.length_factor)

    @overload
    def time_into(self, value: float, units: Units) -> float: ...
    @overload
    def time_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: Units
    ) -> np.ndarray[Any, DT]: ...
    def time_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: Units
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert a time from SI units to the system's units."""
        return value * (self.time_factor / units.time_factor)

    @overload
    def frequency_into(self, value: float, units: Units) -> float: ...
    @overload
    def frequency_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: Units
    ) -> np.ndarray[Any, DT]: ...
    def frequency_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: Units
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert a frequency from SI units to the system's units."""
        return value * (self.time_factor / units.time_factor) ** -1


@dataclass(frozen=True, kw_only=True)
class EtaParameters:
    eta_m: float
    eta_lambda: float
    eta_omega: float
    kbt_div_hbar: float
    units: Units = Units()

    @property
    def temperature(self) -> float:
        return self.units.frequency_into(self.kbt_div_hbar, Units()) * (
            hbar / Boltzmann
        )

    @property
    def m(self) -> float:
        eta_m = self.units.length_into(self.eta_m, Units())
        kb_t_div_hbar = self.units.frequency_into(self.kbt_div_hbar, Units())
        return eta_m * (hbar / 2) / kb_t_div_hbar

    @property
    def lambda_(self) -> float:
        return self.units.frequency_into(self.kbt_div_hbar / self.eta_lambda, Units())

    @property
    def omega(self) -> float:
        return self.units.frequency_into(self.kbt_div_hbar / self.eta_omega, Units())

    @property
    def base_parameters(self) -> BaseParameters:
        return BaseParameters(
            m=self.m,
            lambda_=self.lambda_,
            omega=self.omega,
            temperature=self.temperature,
        )

    def with_units(self, units: Units) -> EtaParameters:
        """Return a copy of the parameters with new units."""
        return EtaParameters(
            eta_m=self.units.length_into(self.kbt_div_hbar, units),
            eta_lambda=self.eta_lambda,
            eta_omega=self.eta_omega,
            kbt_div_hbar=self.units.time_into(self.kbt_div_hbar, units),
            units=units,
        )

    def to_normalized(self) -> EtaParameters:
        """Return a normalized version of the parameters."""
        new_units = Units(length_factor=self.eta_m, time_factor=1 / self.kbt_div_hbar)
        assert self.units.frequency_into(self.kbt_div_hbar, new_units) == 1
        assert self.units.length_into(self.eta_m, new_units) == 1
        out = EtaParameters(
            eta_m=self.units.length_into(self.eta_m, new_units),
            eta_lambda=self.eta_lambda,
            eta_omega=self.eta_omega,
            kbt_div_hbar=self.units.frequency_into(self.kbt_div_hbar, new_units),
            units=new_units,
        )
        assert out.temperature == self.temperature
        return out


@dataclass(frozen=True, kw_only=True)
class BaseParameters:
    m: float
    lambda_: float
    omega: float
    temperature: float

    @property
    def eta_parameters(self) -> EtaParameters:
        kbt_div_hbar = Boltzmann * self.temperature / hbar
        eta_m = kbt_div_hbar / (hbar / (2 * self.m))
        eta_lambda = kbt_div_hbar / self.lambda_
        eta_omega = kbt_div_hbar / self.omega
        return EtaParameters(
            eta_m=eta_m,
            eta_lambda=eta_lambda,
            eta_omega=eta_omega,
            kbt_div_hbar=kbt_div_hbar,
        )

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

    @property
    def omega(self) -> float:
        return self.base_parameters.omega


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
