"""Basic log-space tools and utilities.

This module contains fundamental log-space operations including NaN handling,
log-subtraction, log-sum-exp variants, and normalization functions.
"""

import numpy as np

import fuz.types as ft


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


def lsub(
    la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -np.inf
) -> ft.NPTensor:
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


def complex_lsub(
    la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -np.inf
) -> ft.NPTensor:
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
