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

# TODO(viamiraia): maybe create a logdiffexp function


def fillna(x: ft.Broadcast, nan: ft.Scalar = 0) -> ft.NPTensor:
    """Fill NaN values in an array.

    This function is a faster alternative to ``np.where(np.isnan(x), nan, x)``.

    Parameters
    ----------
    x
        Input array.
    nan
        Value to replace NaNs with.

    Returns
    -------
    ft.NPTensor
        A new array with NaN values replaced.
    """
    x = np.array(x)  # make a copy
    is_nan = np.isnan(x)
    x[is_nan] = nan
    return x


def lsub(la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -np.inf) -> ft.NPTensor:
    """Given log(a) and log(b), calculate log(a-b). Recommend using complex_lsub instead.

    Will fail if a < b. Recommend using :func:`complex_lsub` and taking the real part.

    AKA logsubexp. See
    https://stackoverflow.com/questions/65233445/how-to-calculate-sums-in-log-space-without-underflow

    Parameters
    ----------
    la
        Logarithm of the minuend.
    lb
        Logarithm of the subtrahend.
    nan
        Value to substitute for NaNs in the input arrays.

    Returns
    -------
    ft.NPTensor
        The value of `log(a - b)`.
    """
    la, lb = fillna(la, nan), fillna(lb, nan)
    if la.size > lb.size:
        lb = np.broadcast_to(lb, la.shape)
    elif la.size < lb.size:
        la = np.broadcast_to(la, lb.shape)
    lb_minus_la = lb - la
    method = lb_minus_la < -0.6931471805599453  # noqa: PLR2004

    res = np.empty_like(lb_minus_la)
    res[method] = la[method] + np.log1p(-np.exp(lb_minus_la[method]))

    method = ~method
    res[method] = la[method] + np.log(-np.expm1(lb_minus_la[method]))
    return res


def complex_lsub(la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -np.inf) -> ft.NPTensor:
    """Numerically stable log-space subtraction with complex numbers.

    Calculates `log(a - b)` given `la = log(a)` and `lb = log(b)`, allowing for
    complex results when `a < b`.

    Parameters
    ----------
    la
        Logarithm of the minuend.
    lb
        Logarithm of the subtrahend.
    nan
        Value to substitute for NaNs in the input arrays.

    Returns
    -------
    ft.NPTensor
        The value of `log(a - b)` as a complex number.
    """
    la, lb = fillna(la, nan).astype(complex), fillna(lb, nan).astype(complex)
    if la.size > lb.size:
        lb = np.broadcast_to(lb, la.shape)
    elif la.size < lb.size:
        la = np.broadcast_to(la, lb.shape)
    return la + np.log(-np.expm1(lb - la))


def nanlse(x: ft.Broadcast, axis: int | None = None) -> ft.Broadcast:
    """Get logsumexp treating NaNs as zeroes.

    Parameters
    ----------
    x
        Input array of log-space values.
    axis
        Axis or axes along which to operate.

    Returns
    -------
    ft.Broadcast
        The result of the log-sum-exp operation.
    """
    # copy to avoid side effects
    x = np.array(x)
    c = np.nanmax(x)
    # don't need to handle nans for x-c
    return c + np.log(np.nansum(np.exp(x - c), axis=axis))


def complex_nanlse(x: ft.Broadcast, axis: int | None = None) -> ft.Broadcast:
    """Get logsumexp forcing complex numbers and treating nans as zeroes.

    Parameters
    ----------
    x
        Input array of log-space values.
    axis
        Axis or axes along which to operate.

    Returns
    -------
    ft.Broadcast
        The result of the complex log-sum-exp operation.
    """
    # copy to avoid side effects
    x = np.array(x, dtype=complex)
    c = np.nanmax(x.real)
    return c + np.log(np.nansum(np.exp(x - c), axis=axis))


def limag_sign(x: ft.Broadcast) -> ft.Broadcast:
    """Alternative way of calculating sign for complex numbers.

    Returns an array of the same shape as `x`, with -1 where the imaginary
    component is close to pi, and 1 everywhere else.

    Parameters
    ----------
    x
        Input array of complex numbers.

    Returns
    -------
    ft.Broadcast
        An array of signs.
    """
    x = np.asarray(x, dtype=complex)
    res = np.ones(x.shape)
    near_pi = np.isclose(np.abs(x.imag), np.pi)
    res[near_pi] = -1
    return res


def lnorm(lx: ft.Broadcast, *, force_complex: bool = False) -> ft.Broadcast:
    """Normalize an array in log-space.

    Parameters
    ----------
    lx
        Input array of log-space values.
    force_complex
        If True, forces the calculation to use complex numbers.

    Returns
    -------
    ft.Broadcast
        A normalized log-space array.
    """
    lx = np.array(lx) if not force_complex else np.array(lx, dtype=complex)
    return lx - nanlse(lx) if np.isreal(lx).all() else lx - complex_nanlse(lx)


def norm(x: ft.Broadcast, *, force_complex: bool = False) -> ft.Broadcast:
    """Normalize an array in linear space.

    This is a convenience wrapper around :func:`lnorm`.

    Parameters
    ----------
    x
        Input array of linear-space values.
    force_complex
        If True, forces the calculation to use complex numbers.

    Returns
    -------
    ft.Broadcast
        A normalized linear-space array.
    """
    return np.exp(lnorm(np.log(x), force_complex=force_complex))


def log_trap(
    fx: np.ndarray, x_or_dx: ft.Scalar | np.ndarray, xax: Literal[0, 1] = 1
) -> np.ndarray:
    """Given log function values, get the log integral using log trapezoidal integration.

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
    return nanlse(np.log(x[1:] - x[:-1]) - np.log(2) + np.logaddexp(fx[:-1], fx[1:]))


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
    to_sum = np.array((fx[:-1:3], fx[1::3] + np.log(3), fx[2::3] + np.log(3), fx[3::3]))
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
