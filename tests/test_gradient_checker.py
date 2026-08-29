import numpy as np
import numpy.typing as npt
import pytest

from gradlab.gradient_checker import compare_gradients, compute_numerical_gradient

ArrayFloat64 = npt.NDArray[np.float64]
ArrayInt64 = npt.NDArray[np.int64]

TOLERANCE_FLOAT64: float = 1e-7


def test_exact_analytical_gradient() -> None:
    """Nominal test: verifies that the numerical gradient matches the exact analytical gradient."""

    def f(x: ArrayFloat64) -> float:
        return float(np.sum(x**3))

    x: ArrayFloat64 = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    grad_analytical: ArrayFloat64 = 3.0 * (x**2)

    grad_numerical: ArrayFloat64 = compute_numerical_gradient(f, x)
    error: float = compare_gradients(grad_analytical, grad_numerical)

    assert error < TOLERANCE_FLOAT64


def test_two_parameters_function_different_shapes() -> None:
    """
    Edge case/collision test: evaluates the gradient of a SINGLE function depending
    on two parameters of different shapes, isolating the calculation via closures.
    """
    # Single Source of Truth definition to avoid duplication
    inputs: ArrayFloat64 = np.array([0.5, 1.5], dtype=np.float64)
    w_init: ArrayFloat64 = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
    )
    b_init: ArrayFloat64 = np.array([-1.0, 1.0, 0.5], dtype=np.float64)

    # The target function correctly takes both parameters
    def f_mult(w: ArrayFloat64, b: ArrayFloat64) -> float:
        return float(np.sum(w @ inputs + b))

    # To test W, we freeze b. To test b, we freeze W.
    grad_w: ArrayFloat64 = compute_numerical_gradient(
        lambda w: f_mult(w, b_init), w_init
    )
    grad_b: ArrayFloat64 = compute_numerical_gradient(
        lambda b: f_mult(w_init, b), b_init
    )

    grad_w_analytical: ArrayFloat64 = np.array(
        [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]], dtype=np.float64
    )
    grad_b_analytical: ArrayFloat64 = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    assert compare_gradients(grad_w_analytical, grad_w) < TOLERANCE_FLOAT64
    assert compare_gradients(grad_b_analytical, grad_b) < TOLERANCE_FLOAT64


def test_shape_preserved_2_3() -> None:
    """Shape test: the gradient array must have the same shape as the input."""

    def f(x: ArrayFloat64) -> float:
        return float(np.sum(np.sin(x)))

    x: ArrayFloat64 = np.arange(6, dtype=np.float64).reshape(2, 3)
    grad: ArrayFloat64 = compute_numerical_gradient(f, x)

    assert grad.shape == x.shape


def test_checker_detects_wrong_gradient() -> None:
    """
    Test proving the tool works: an injected 1% relative error
    must be measured by the comparator and rejected.
    """

    def f(x: ArrayFloat64) -> float:
        return float(np.sum(x**2))

    x: ArrayFloat64 = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    grad_analytical_true: ArrayFloat64 = 2.0 * x
    grad_analytical_false: ArrayFloat64 = grad_analytical_true * 1.01

    grad_numerical: ArrayFloat64 = compute_numerical_gradient(f, x)
    relative_error: float = compare_gradients(grad_analytical_false, grad_numerical)

    assert relative_error > 0.009


def test_input_not_modified() -> None:
    """What must not happen: the function must NEVER mutate the input array."""

    def f(x: ArrayFloat64) -> float:
        return float(np.sum(x**4))

    x_original: ArrayFloat64 = np.array([1.0, -2.5, 3.1415], dtype=np.float64)
    x_copy: ArrayFloat64 = x_original.copy()

    _ = compute_numerical_gradient(f, x_original)

    np.testing.assert_array_equal(x_original, x_copy)


def test_comparison_division_by_zero() -> None:
    """Test of the comparison function against zero gradients."""
    grad1: ArrayFloat64 = np.zeros(5, dtype=np.float64)
    grad2: ArrayFloat64 = np.zeros(5, dtype=np.float64)

    error: float = compare_gradients(grad1, grad2)
    assert error == 0.0


def test_invalid_inputs_reject_integers() -> None:
    """
    Test for invalid inputs: silently adding a float (epsilon)
    to an integer array causes a catastrophic implicit cast.
    Architectural choice: the function MUST raise a clear TypeError.
    """

    def f(x: ArrayFloat64) -> float:
        return float(np.sum(x**2))

    x_int: ArrayInt64 = np.array([1, 2, 3], dtype=np.int64)

    with pytest.raises(TypeError, match="float64"):
        # We intentionally ignore the type incompatibility expected by mypy to test the runtime
        compute_numerical_gradient(f, x_int)  # type: ignore[arg-type]


def test_compare_gradients_incompatible_shapes() -> None:
    """Verifies that compare_gradients raises a TypeError in case of incompatible shapes."""
    grad_analy: ArrayFloat64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    grad_numeric: ArrayFloat64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="the shape must be the same"):
        compare_gradients(grad_analy, grad_numeric)
