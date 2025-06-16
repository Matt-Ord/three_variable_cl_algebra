# If we have an SSE of the form d|psi> = A |psi> dt + B |psi> dW, we can
# identify the most stable state as the eigenstate of the operator A
# with the lowest eigenvalue.
# If we want to find the most stable state at a particular position (alpha)
# we can instead find the eigenstates of the operator A - mu B.
# For the harmonic potential, we should find that these eigenstates
# match with our prediction from the two variable model.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import Boltzmann  # type: ignore lib
from scipy.sparse.linalg import LinearOperator, lgmres
from slate_core import linalg
from slate_quantum import State, operator

from three_variable.simulation import get_condition_from_params

if TYPE_CHECKING:
    from collections.abc import Callable

    from adsorbate_simulation.system import SimulationCondition
    from slate_core import Basis

condition = get_condition_from_params(
    eta_omega=1.0,
    eta_lambda=1.0,
    mass=1.0,
    temperature=10 / Boltzmann,
    minimum_occupation=1e-6,
    truncate=True,
)


def build_linear_operator(
    shape: tuple[int, int],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.complexfloating]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> LinearOperator:
    """Build a linear operator for the system."""
    return LinearOperator(shape=shape, matvec=fn)  # type: ignore[return-value]


def get_deterministic_environment_operator_fn(
    condition: SimulationCondition, state_basis: Basis
) -> Callable[
    [np.ndarray[Any, np.dtype[np.complexfloating]]],
    np.ndarray[Any, np.dtype[np.complexfloating]],
]:
    hamiltonian = condition.hamiltonian
    shift_operator = operator.build.caldeira_leggett_shift(
        hamiltonian.basis.metadata().children[0], friction=condition.gamma
    )
    collapse_operator = operator.build.caldeira_leggett_collapse(
        hamiltonian.basis.metadata().children[0],
        friction=condition.gamma,
        temperature=condition.temperature,
        mass=condition.mass,
    )

    def _fn(x: np.ndarray[Any, np.dtype[np.complexfloating]]):
        """Function to compute the deterministic environment operator."""
        state = State(state_basis, x)
        # The deterministic environment operator is given by
        # A = -i / hbar (H + H') + <L dagger>L - 0.5 * L dagger L - 0.5 * <L dagger> * <L> |psi>
        # TODO: better __sub__ support in Slate
        expectation_value = operator.expectation(collapse_operator, state)
        environment_operator = (
            collapse_operator * np.conjugate(expectation_value)
            - operator.matmul(
                operator.dagger(collapse_operator),
                (collapse_operator * 0.5j).as_type(np.complex128),
            )
            - expectation_value * (0.5 * np.conjugate(expectation_value))
        )
        a = hamiltonian + shift_operator + environment_operator
        # print(a.basis.metadata())
        # print(state_basis.metadata())

        assert a.basis.metadata().children[1] == state_basis.metadata(), (
            "Basis mismatch in environment operator"
        )
        return (
            linalg.einsum(
                "(i j'),j->i",
                hamiltonian + shift_operator + environment_operator,
                state,
            )
            .with_basis(state_basis)
            .raw_data
        )

    return _fn


def get_deterministic_environment_operator(
    condition: SimulationCondition, state_basis: Basis
) -> LinearOperator:
    return build_linear_operator(
        shape=(state_basis.size, state_basis.size),
        fn=get_deterministic_environment_operator_fn(condition, state_basis),
    )


def get_pointer_state(condition: SimulationCondition[Any, Any]) -> State:
    """Get the pointer state for the system."""
    initial_state = condition.initial_state

    pointer_state_data = lgmres(
        get_deterministic_environment_operator(
            condition,
            initial_state.basis,
        ),b = ,
        b=initial_state.raw_data,
        rtol=1e-6,
        maxiter=1000,
    )

    return State(initial_state.basis, pointer_state_data[0])


# The issue - A contains terms like L - <L> |psi> which are not proper operators.
print(get_pointer_state(condition))
