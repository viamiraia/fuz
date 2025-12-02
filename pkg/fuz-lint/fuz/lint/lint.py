"""Log-space numerical integration.

This module provides numerically-stable integration primitives that operate
directly in log-space (i.e. they work with `ln(x)` instead of `x`).

The main integration functions are:

* ``ltrap``: Log-space trapezoidal integration (scipy.integrate.trapezoid-like)
* ``lsimp``: Log-space Simpson's rule integration (scipy.integrate.simpson-like)
* ``lsimp13``: Log-space Simpson's 1/3 rule for uniformly spaced data
* ``lsimp38``: Log-space Simpson's 3/8 rule for uniformly spaced data
* ``lsimp_irreg``: Log-space Simpson's rule for irregularly spaced data

All functions accept arrays that can be converted to numpy arrays and return
numpy arrays of matching dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fuz.lint.ltools import (
    complex_lsub,
    complex_nanlse,
    fillna,
    limag_sign,
    lnorm,
    lsub,
    nanlse,
    norm,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_MIN_SIMPSON_POINTS = 3

__all__ = [
    # Re-exports from ltools
    'fillna',
    'lsub',
    'complex_lsub',
    'nanlse',
    'complex_nanlse',
    'limag_sign',
    'lnorm',
    'norm',
    # Integration functions
    'ltrap',
    'lsimp',
    'lsimp13',
    'lsimp38',
    'lsimp_irreg',
]


def ltrap(
    y: ArrayLike,
    x: ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> NDArray[np.floating]:
    """Log-space trapezoidal integration.

    Integrate y(x) in log-space using the composite trapezoidal rule.
    This computes log(integral(exp(y))) in a numerically stable way.

    Parameters
    ----------
    y
        Logarithm of function values to integrate.
    x
        The sample points corresponding to the y values. If None, the sample
        points are assumed to be evenly spaced dx apart.
    dx
        The spacing between sample points when x is None. Default is 1.0.
    axis
        The axis along which to integrate. Default is -1 (last axis).

    Returns
    -------
    NDArray[np.floating]
        The logarithm of the definite integral.

    See Also
    --------
    scipy.integrate.trapezoid : Linear-space trapezoidal integration.

    Examples
    --------
    >>> import numpy as np
    >>> from fuz.lint import ltrap
    >>> # Integrate exp(y) where y = log(x^2) from 0.01 to 1
    >>> x = np.linspace(0.01, 1, 100)
    >>> y = 2 * np.log(x)  # log(x^2)
    >>> result = ltrap(y, x=x)
    >>> np.exp(result)  # Should be close to 1/3
    0.333...
    """
    y = np.asarray(y)

    # Move the integration axis to the last position for consistent processing
    y = np.moveaxis(y, axis, -1)

    # Compute log of trapezoid weights: log((y[i] + y[i+1]) / 2 * dx)
    # = logaddexp(y[i], y[i+1]) - log(2) + log(dx)
    pairwise_sum = np.logaddexp(y[..., :-1], y[..., 1:])

    if x is not None:
        x = np.asarray(x)
        # Non-uniform spacing: log(dx_i) for each interval
        log_dx = np.log(np.abs(np.diff(x)))
        to_sum = pairwise_sum + log_dx - np.log(2)
    else:
        # Uniform spacing
        to_sum = pairwise_sum + np.log(dx) - np.log(2)

    return nanlse(to_sum, axis=-1)


def lsimp(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> NDArray[np.floating]:
    """Log-space Simpson's rule integration.

    Integrate y(x) in log-space using Simpson's rule. This computes
    log(integral(exp(y))) in a numerically stable way.

    For uniformly spaced data with an odd number of points, uses Simpson's 1/3
    rule. For non-uniformly spaced data or even number of points, uses the
    irregular Simpson's rule.

    Parameters
    ----------
    y
        Logarithm of function values to integrate.
    x
        The sample points corresponding to the y values. If None, the sample
        points are assumed to be evenly spaced dx apart.
    dx
        The spacing between sample points when x is None. Default is 1.0.
    axis
        The axis along which to integrate. Default is -1 (last axis).

    Returns
    -------
    NDArray[np.floating]
        The logarithm of the definite integral.

    See Also
    --------
    scipy.integrate.simpson : Linear-space Simpson's rule integration.
    lsimp13 : Log-space Simpson's 1/3 rule (uniformly spaced, odd points).
    lsimp38 : Log-space Simpson's 3/8 rule (uniformly spaced, n%3==0 intervals).
    lsimp_irreg : Log-space Simpson's rule for irregular spacing.

    Examples
    --------
    >>> import numpy as np
    >>> from fuz.lint import lsimp
    >>> x = np.linspace(0, 1, 101)  # odd number of points
    >>> y = np.log(x**2 + 1)  # log of function values
    >>> result = lsimp(y, x=x)
    """
    y = np.asarray(y)

    # Move the integration axis to the last position
    y = np.moveaxis(y, axis, -1)
    n = y.shape[-1]

    if x is not None:
        x = np.asarray(x)
        # Check if uniformly spaced
        diffs = np.diff(x)
        is_uniform = np.allclose(diffs, diffs[0], rtol=1e-10)

        if is_uniform and n % 2 == 1 and n >= _MIN_SIMPSON_POINTS:
            # Use Simpson's 1/3 rule for uniform spacing with odd points
            return _lsimp13_impl(y, x)
        # Use irregular Simpson's rule
        return _lsimp_irreg_impl(y, x)
    # Uniform spacing with dx
    x = np.arange(n, dtype=float) * dx
    if n % 2 == 1 and n >= _MIN_SIMPSON_POINTS:
        return _lsimp13_impl(y, x)
    return _lsimp_irreg_impl(y, x)


def _lsimp13_impl(y: np.ndarray, x: np.ndarray) -> np.floating:
    """Compute Simpson's 1/3 rule in log-space.

    Simpson's 1/3 rule: integral ≈ (dx/3) * (f0 + 4*f1 + 2*f2 + 4*f3 + ... + fn)
    Weights: 1, 4, 2, 4, 2, ..., 4, 1
    """
    n = x.size
    ldx = np.log(x[-1] - x[0]) - np.log(n - 1)

    # Build weights array: 1, 4, 2, 4, 2, ..., 4, 1
    weights = np.ones(n)
    weights[1:-1:2] = 4  # odd indices (1, 3, 5, ...)
    weights[2:-1:2] = 2  # even interior indices (2, 4, 6, ...)

    log_weighted = y + np.log(weights)
    return ldx - np.log(3) + nanlse(log_weighted, axis=-1)


def _lsimp_irreg_impl(y: np.ndarray, x: np.ndarray) -> np.floating:
    """Compute irregular Simpson's rule in log-space."""
    n = x.size - 1
    h = x[1:] - x[:-1]

    # Process pairs of intervals (up to the last complete pair)
    # For n intervals, we process floor(n/2) pairs
    n_pairs = n // 2

    if n_pairs == 0:
        # Only one interval - use trapezoidal rule
        return np.log(h[0]) - np.log(2) + np.logaddexp(y[0], y[1])

    # Get intervals for complete pairs
    h0 = h[: 2 * n_pairs : 2]  # even intervals: 0, 2, 4, ...
    h1 = h[1 : 2 * n_pairs : 2]  # odd intervals: 1, 3, 5, ...

    hdh = h1 / h0
    hmh = h1 * h0
    lhph = np.log(h1 + h0)

    # Handle potential numerical issues with coefficients
    coef0 = 2 - hdh
    coef2 = 2 - 1 / hdh

    # Use absolute values for log computation
    log_coef0 = np.log(np.abs(coef0))
    log_coef2 = np.log(np.abs(coef2))

    # Get y values for complete pairs
    y0 = y[: 2 * n_pairs : 2]  # y[0], y[2], y[4], ...
    y1 = y[1 : 2 * n_pairs : 2]  # y[1], y[3], y[5], ...
    y2 = y[2 : 2 * n_pairs + 1 : 2]  # y[2], y[4], y[6], ...

    # Compute contribution from each pair
    pair_contrib = nanlse(
        np.array(
            [
                lhph + log_coef0 + y0,
                3 * lhph - np.log(hmh) + y1,
                lhph + log_coef2 + y2,
            ]
        ),
        axis=0,
    ) - np.log(6)

    # Sum all pair contributions
    res = nanlse(pair_contrib)

    if n % 2 == 1:
        # Handle the last interval with trapezoidal-like correction
        h0_last, h1_last = h[n - 2], h[n - 1]
        h1sq = h1_last**2
        hmh3 = 3 * h1_last * h0_last
        hph6 = (h1_last + h0_last) * 6

        term1 = y[n] + np.log(np.abs((2 * h1sq + hmh3) / hph6))
        term2 = y[n - 1] + np.log(np.abs((h1sq + hmh3) / (6 * h0_last)))
        term3 = y[n - 2] + np.log(np.abs(h1sq * h1_last / (h0_last * hph6)))

        res = nanlse([res, term1, term2, term3])

    return res


def lsimp13(y: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """Log-space Simpson's 1/3 rule for numerical integration.

    Parameters
    ----------
    y
        Logarithm of function values to integrate. Must have an odd number of
        points (at least 3).
    x
        Array of x-coordinates, assumed to be uniformly spaced.

    Returns
    -------
    NDArray[np.floating]
        The logarithm of the definite integral.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    return _lsimp13_impl(y, x)


def lsimp38(y: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """Log-space Simpson's 3/8 rule for numerical integration.

    Parameters
    ----------
    y
        Logarithm of function values to integrate. The number of intervals must
        be a multiple of 3 (i.e., n-1 must be divisible by 3).
    x
        Array of x-coordinates, assumed to be uniformly spaced.

    Returns
    -------
    NDArray[np.floating]
        The logarithm of the definite integral.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    n = x.size
    ldx = np.log(x[-1] - x[0]) - np.log(n - 1)  # Fixed: was np.log(n)

    # Simpson's 3/8 weights: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
    weights = np.ones(n)
    weights[1::3] = 3
    weights[2::3] = 3
    weights[3:-1:3] = 2

    log_weighted = y + np.log(weights)
    return ldx - np.log(8) + np.log(3) + nanlse(log_weighted)


def lsimp_irreg(y: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """Log-space Simpson's rule for irregularly spaced data.

    Parameters
    ----------
    y
        Logarithm of function values to integrate over.
    x
        Array of x-coordinates, which can be irregularly spaced.

    Returns
    -------
    NDArray[np.floating]
        The logarithm of the integral.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    return _lsimp_irreg_impl(y, x)
