from __future__ import annotations

import numpy as np

from .equilibrium_squeeze import evaluate_equilibrium_R


def get_beta_from_ratio(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    beta_ratio = evaluate_equilibrium_R(eta_m, eta_omega, eta_lambda, positive=positive)
    return (1 - beta_ratio) / (1 + beta_ratio)  # type: ignore cant infer shape


def get_beta_0(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""
    Get the beta_0 operator.

    .. math::
        \beta_0 = \frac{1 - 2 \eta_m}{1 + 2 \eta_m}

    Returns
    -------
        np.ndarray: the value of beta_0
    """
    return (1 + 2 * eta_m) / (1 - 2 * eta_m)  # type: ignore cant infer shape


def get_beta_1_fraction(
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    r"""
    Get the beta_1 squared operator.

    .. math::
        (\beta_1 - 1 / \beta_1)^2 = \frac{4i\eta_\omega^2}{ \eta_\lambda (4 \eta_\omega^2 - 1)}

    Returns
    -------
        np.ndarray: the value of beta_1 squared
    """
    return (4j * eta_omega**2) / (eta_lambda * (4 * eta_omega**2 - 1))  # type: ignore cant infer shape


def get_beta_1(
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = True,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    r"""
    Get the beta_1 operator.

    .. math::
        \beta_1 = 1 +- \sqrt{\frac{16i\eta \frac{\lambda}{\omega}}{16 \eta^2 - 1}}

    Returns
    -------
        np.ndarray: the value of beta_1
    """
    sqrt_val = np.emath.sqrt(get_beta_1_fraction(eta_omega, eta_lambda))
    denominator = 1 - sqrt_val if positive else 1 + sqrt_val
    return 1 / denominator  # type: ignore cant infer shape


def get_beta(
    eta_m: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_omega: np.ndarray[tuple[int], np.dtype[np.float64]],
    eta_lambda: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    positive: bool = True,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    r"""
    Get the beta operator.

    .. math::
        \beta = \frac{\beta_0 + \beta_1}{1 + \beta_0 \beta_1}

    """
    beta_0 = get_beta_0(eta_m)
    beta_1 = get_beta_1(eta_omega, eta_lambda, positive=positive)
    return (beta_0 + beta_1) / (1 + beta_0 * beta_1)  # type: ignore cant infer shape


def get_joint_uncertainty(
    beta: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""
    Get the joint uncertainty operator.

    .. math::
        \Delta x \Delta p = \frac{1}{2} \left| \frac{1 - \beta}{1 + \beta} \right|

    """
    prefactor = 1 / (4 * (1 - np.abs(beta) ** 2))
    error_1 = 1 + np.abs(beta) ** 2 + 2 * np.real(beta)
    error_2 = 1 + np.abs(beta) ** 2 - 2 * np.real(beta)
    return prefactor**2 * (error_1 * error_2)  # type: ignore cant infer shape


def get_joint_uncertainty_from_angle(
    beta: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""
    Get the joint uncertainty operator.

    .. math::
        \Delta x \Delta p = \frac{1}{2} \left| \frac{1 - \beta}{1 + \beta} \right|

    """
    r = np.atanh(np.abs(beta))
    theta = np.angle(beta)

    mu = np.cosh(r)
    nu = np.sinh(r)

    prefactor = 1 / 4
    error_1 = mu**2 + nu**2 - 2 * mu * nu * np.cos(theta)
    error_2 = mu**2 + nu**2 + 2 * mu * nu * np.cos(theta)
    return prefactor**2 * (error_1 * error_2)  # type: ignore cant infer shape
