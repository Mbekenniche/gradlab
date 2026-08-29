from collections.abc import Callable

import numpy as np
import numpy.typing as npt

EPSILON: float = 1e-5
"""
Step size for the numerical gradient computation.

The value 1e-5 is optimal for float64 precision. Since the error of the
centered difference gradient is O(epsilon^2), this value balances the
theoretical truncation error and rounding noise, leaving a relative precision
on the order of 1e-10 to 1e-8.
"""

DENOMINATOR_EPSILON: float = 1e-12

ArrayFloat64 = npt.NDArray[np.float64]


def compute_numerical_gradient(
    f: Callable[[ArrayFloat64], float],
    x: ArrayFloat64,
) -> ArrayFloat64:
    if x.dtype != np.float64:
        raise TypeError("The dtype must be float64")

    gradient: ArrayFloat64 = np.zeros_like(x, dtype=np.float64)
    x_copy: ArrayFloat64 = x.copy()
    it = np.nditer(x_copy, flags=["multi_index"])

    for _ in it:
        idx = it.multi_index
        i_keep = float(x_copy[idx])

        x_copy[idx] = i_keep + EPSILON
        i_pplus = f(x_copy)

        x_copy[idx] = i_keep - EPSILON
        i_pminus = f(x_copy)

        x_copy[idx] = i_keep

        gap = (i_pplus - i_pminus) / (2 * EPSILON)
        gradient[idx] = gap

    return gradient


def compare_gradients(analytical: ArrayFloat64, numeric: ArrayFloat64) -> float:
    if analytical.shape != numeric.shape:
        raise ValueError(
            "the shape must be the same. analytical.shape != numeric.shape"
        )
    norm_diff = float(np.linalg.norm(analytical - numeric))
    max_norm = (
        max(float(np.linalg.norm(analytical)), float(np.linalg.norm(numeric)))
        + DENOMINATOR_EPSILON
    )
    return norm_diff / max_norm
