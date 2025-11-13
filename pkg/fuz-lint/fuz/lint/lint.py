"""Log-space numerical helpers.

This module collects numerically-stable primitives that operate directly in
log-space (i.e. they work with `ln(x)` instead of `x`).  The helpers are grouped
into four broad categories:

* Basic element-wise manipulations: ``fillna`` and the log-subtraction helpers
  ``lsub`` / ``complex_lsub``.
* Log-sum-exp variants that gracefully skip ``NaN`` values: ``nanlse`` and
  ``complex_nanlse``.
* Probability normalisation: ``lnorm`` and its exponential counterpart
  ``norm``.
* Log-space numerical integration schemes: ``log_trap``, ``ltrap``,
  ``lsimp13``, ``lsimp38``, and ``lsimp_irreg``.

All public helpers obey the type aliases defined in :pymod:`fuz.types` and aim
for a balance between speed, clarity, and allocation-free operation.  Unless
otherwise stated the functions accept any object that can be losslessly
converted to :class:`~numpy.ndarray` via ``numpy.asarray`` and return a
:class:`~numpy.ndarray` (or a broadcast-compatible scalar) of matching dtype.
"""

from typing import Literal

import numpy as np
from plum import dispatch

import fuz.types as ft
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

# TODO(viamiraia): maybe create a logdiffexp function

# Re-export the functions from ltools for backwards compatibility
__all__ = [
    'fillna',
    'lsub',
    'complex_lsub',
    'nanlse',
    'complex_nanlse',
    'limag_sign',
    'lnorm',
    'norm',
    'log_trap',
    'ltrap',
    'lsimp13',
    'lsimp38',
    'lsimp_irreg',
]


def log_trap(
    fx: np.ndarray, x_or_dx: ft.Scalar | np.ndarray, xax: Literal[0, 1] = 1
) -> np.ndarray:
    """Given log function values, get log integral using log trapezoidal integration.

    Can perform vectorized integration up to two dimensions.

    Parameters
    ----------
    fx
        Logarithm of function values to integrate over.
    x_or_dx
        Either a scalar representing the uniform spacing (dx) between points, or a
        1D array of x-coordinates.
    xax
        The axis corresponding to x in fx. Default is 1.

    Returns
    -------
    np.ndarray
        The logarithm of the integral.

    Notes
    -----
    Using scipy's logsumexp b parameter is slower than doing the scaling yourself.
    Dividing x_or_dx by 2 first is a tiny bit faster but might be less stable.
    """
    match xax, len(fx.shape):
        case 1, 1:
            to_sum = np.logaddexp(fx[:-1], fx[1:]).reshape(1, -1)
        case 0, 1:
            to_sum = np.logaddexp(fx[:-1], fx[1:]).reshape(-1, 1)
        case 1, _:
            to_sum = np.logaddexp(fx[:, :-1], fx[:, 1:])
        case _, _:
            to_sum = np.logaddexp(fx[:-1, :], fx[1:, :])

    x_is_vec = isinstance(x_or_dx, np.ndarray) and x_or_dx.size > 1
    match xax, x_is_vec:
        case 1, True:
            to_sum += np.log(np.diff(x_or_dx)).reshape(1, -1)
        case 0, True:
            to_sum += np.log(np.diff(x_or_dx)).reshape(-1, 1)
        case _, False:
            return nanlse(to_sum, axis=xax) - np.log(2) + np.log(x_or_dx)
    return nanlse(to_sum, axis=xax) - np.log(2)


@dispatch
def ltrap(fx: np.ndarray, dx: float) -> float:
    """Log-space trapezoidal integration for uniformly spaced data.

    This is a specialized version of :func:`log_trap` for uniformly spaced data.

    Parameters
    ----------
    fx
        Logarithm of function values to integrate.
    dx
        The uniform spacing between points.

    Returns
    -------
    float
        The logarithm of the definite integral.
    """
    return np.log(dx) - np.log(2) + nanlse(np.array((fx[:-1], fx[1:])))


@dispatch
def ltrap(fx: np.ndarray, x: np.ndarray) -> float:
    """Log-space trapezoidal integration for non-uniformly spaced data.

    This is a specialized version of :func:`log_trap` for non-uniformly spaced
    data.

    Parameters
    ----------
    fx
        Logarithm of function values to integrate.
    x
        An array of x-coordinates.

    Returns
    -------
    float
        The logarithm of the definite integral.
    """
    return nanlse(
        np.log(x[1:] - x[:-1]) - np.log(2) + np.logaddexp(fx[:-1], fx[1:])
    )


def lsimp13(fx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Log-space Simpson's 1/3 rule for numerical integration.

    Parameters
    ----------
    fx
        Logarithm of function values to integrate. Must have an odd number of
        points.
    x
        Array of x-coordinates, assumed to be uniformly spaced.

    Returns
    -------
    np.ndarray
        The logarithm of the definite integral.
    """
    n = x.size
    ldx = np.log(x[-1] - x[0]) - np.log(n - 1)
    to_sum = np.array((fx[:-1:2], fx[1::2] + np.log(4), fx[2::2]))
    return ldx - np.log(3) + nanlse(to_sum)


def lsimp38(fx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Log-space Simpson's 3/8 rule for numerical integration.

    Parameters
    ----------
    fx
        Logarithm of function values to integrate. The number of intervals must
        be a multiple of 3.
    x
        Array of x-coordinates, assumed to be uniformly spaced.

    Returns
    -------
    np.ndarray
        The logarithm of the definite integral.
    """
    n = x.size
    ldx = np.log(x[-1] - x[0]) - np.log(n)
    to_sum = np.array(
        (fx[:-1:3], fx[1::3] + np.log(3), fx[2::3] + np.log(3), fx[3::3])
    )
    return ldx - np.log(8) + np.log(3) + nanlse(to_sum)


def lsimp_irreg(fx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Log-space Simpson's rule for irregularly spaced data.

    Parameters
    ----------
    fx
        Logarithm of function values to integrate over.
    x
        Array of x-coordinates, which can be irregularly spaced.

    Returns
    -------
    np.ndarray
        The logarithm of the integral.
    """
    n = x.size - 1
    h = x[1:] - x[:-1]

    h0, h1 = h[:-1:2], h[1::2]
    hdh, hmh = h1 / h0, h1 * h0
    lhph = np.log(h1 + h0)
    res = nanlse(
        [
            lhph + np.log(2 - hdh) + fx[:-2:2],
            3 * lhph - np.log(hmh) + fx[1:-1:2],
            lhph + np.log(2 - 1 / hdh) + fx[2::2],
        ]
    ) - np.log(6)

    if n % 2 == 1:
        h0, h1 = h[n - 2], h[n - 1]
        h1sq = h1**2
        hmh3 = 3 * h1 * h0
        hph6 = (h1 + h0) * 6
        res = nanlse(
            [
                res,
                fx[n] + np.log((2 * h1sq + hmh3) / hph6),
                fx[n - 1] + np.log((h1sq + hmh3) / (6 * h0)),
                np.log(np.exp(fx[n - 2]) * h1sq * h1 / (h0 * hph6)),
            ]
        )
    return res
