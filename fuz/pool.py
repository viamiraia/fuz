"""Core functions for probability fusion."""

# TODO(viamiraia): Completely refactor / simplify / vectorize.
# Sample pooling and RV pooling are very similar. Try dispatch.

from collections.abc import Sequence
from functools import partial, wraps
from typing import Literal

import numpy as np
from plum import Dispatcher, overload
from scipy.integrate import quad, simpson

import fuz.log as fl
import fuz.types as ft

dispatch = Dispatcher()

dist_weights = wraps(fl.lnorm)
dist_weights.__doc__ = 'Make weights sum to 1. Alias for `fuz.log.lnorm`.'


"""
================================================================================================
rv_continuous-based functions
================================================================================================
"""

""" RV-based closures to handle weights """


def num_f_geom(rvs: ft.ContDists) -> ft.PoolNumF:
    """Closure to get geometric pooling numerator function for SciPy Continuous RVs."""
    return lambda x: np.sum([rv.logpdf(x) for rv in rvs]) / len(rvs)


def num_f_mult(rvs: ft.ContDists) -> ft.PoolNumF:
    """Closure to get multiplicative pooling numerator function for SciPy Continuous RVs."""
    return lambda x: np.sum([rv.logpdf(x) for rv in rvs])


def num_f_weighted(rvs: ft.ContDists, weights: ft.NPVec) -> ft.PoolNumF:
    """Closure to get weighted pooling numerator function for SciPy Continuous RVs."""
    zipped = zip(rvs, weights, strict=True)
    return lambda x: np.sum([w * rv.logpdf(x) for rv, w in zipped])


def _get_norm_const(bounds: tuple[float, float], num_f: ft.PoolNumF) -> float:
    """Gets pooling normalizing constant for SciPy Continuous RVs."""
    integrate_lo, integrate_hi = bounds
    return quad(lambda x: np.exp(num_f(x)), integrate_lo, integrate_hi)[0]


get_norm_const = partial(_get_norm_const, (0, 1))
get_norm_const.__doc__ = 'Gets pooling normalizing constant for continuous belief RVs.'


# TODO(viamiraia): test performance of pool_rv_f implementations


def pool_rv_f(num_f: ft.PoolNumF) -> ft.PoolF:
    """Returns a pooling function, bounded 0-1, given a continuous numerator function."""
    norm_const = get_norm_const(num_f)
    # Instead of adding log(1/const), subtract log(const)
    return lambda x: np.exp(num_f(x) - np.log(norm_const))


def _pool_rv_f(bounds: tuple[float, float], num_f: ft.PoolNumF) -> ft.PoolF:
    """Returns a custom-bounded pooling function given a numerator function."""
    norm_const = _get_norm_const(bounds, num_f)
    return lambda x: np.exp(num_f(x) - np.log(norm_const))


""" Custom bounds pooling """


def _geom_pool(bounds: tuple[float, float], x: ft.NPVec, rvs: ft.ContDists) -> ft.NPVec:
    """Performs custom-bounded geometric pooling for continuous RVs."""
    return _pool_rv_f(bounds, num_f_geom(rvs))(x)


def _mult_pool(bounds: tuple[float, float], x: ft.NPVec, rvs: ft.ContDists) -> ft.NPVec:
    """Performs custom-bounded multiplicative pooling for continuous RVs."""
    return _pool_rv_f(bounds, num_f_mult(rvs))(x)


def _weighted_pool(
    bounds: tuple[float, float],
    x: ft.NPVec,
    rvs: ft.ContDists,
    weights: ft.NPVec,
) -> ft.NPVec:
    """Performs custom-bounded weighted pooling for continuous RVs."""
    return _pool_rv_f(bounds, num_f_weighted(rvs, weights))(x)


""" Bounded 0-1 pooling """


# TODO(viamiraia): test if partial implementation affects performance


def geom_pool(x: ft.NPVec, rvs: ft.ContDists) -> ft.NPVec:
    """Performs geometric pooling for continuous RVs between 0 and 1."""
    return pool_rv_f(num_f_geom(rvs))(x)


def mult_pool(x: ft.NPVec, rvs: ft.ContDists) -> ft.NPVec:
    """Performs multiplicative pooling for continuous RVs between 0 and 1."""
    return pool_rv_f(num_f_mult(rvs))(x)


def weighted_pool(x: ft.NPVec, rvs: ft.ContDists, weights: ft.NPVec) -> ft.NPVec:
    """Performs weighted pooling for continuous RVs between 0 and 1.

    Weights should add up to 1.
    """
    return pool_rv_f(num_f_weighted(rvs, weights))(x)


"""
================================================================================================
Sample-based pooling
================================================================================================
"""

""" Sample-based closures to handle weights """


def num_f_geom_s(pdf_s: Sequence[ft.DistVec]) -> ft.PoolNumFS:
    """Closure to get geometric pooling numerator function for sampled RVs."""
    return lambda: np.sum([np.log(sample) for sample in pdf_s]) / len(pdf_s)


def num_f_mult_s(pdf_s: Sequence[ft.DistVec]) -> ft.PoolNumFS:
    """Closure to get multiplicative pooling numerator function for sampled RVs."""
    return lambda: np.sum([np.log(sample) for sample in pdf_s])


def num_f_weighted_s(pdf_s: Sequence[ft.DistVec], weights: ft.NPVec) -> ft.PoolNumFS:
    """Closure to get weighted pooling numerator function for sampled RVs."""
    zipped = zip(pdf_s, weights, strict=True)  # for type narrowing compatibility
    return lambda: np.sum([w * np.log(sample) for sample, w in zipped])


def get_norm_const_s(lnumerator: ft.NPVec, x: ft.NPVec) -> ft.NPVec:
    """Gets the normalizing constant for sampled RVs using Simpson's rule."""
    return fl.lsimp_irreg(lnumerator, x)


def pool_samples_f(num_f_s: ft.PoolNumF) -> ft.PoolFS:
    """Returns a pooling function given a sample-based numerator function."""
    numerator = num_f_s()
    return lambda x: np.exp(numerator - np.log(get_norm_const_s(numerator, x)))


""" Sample-based pooling functions """


def geom_pool_s(x: ft.NPVec, pdf_samples: Sequence[ft.DistVec]) -> ft.NPVec:
    """Performs geometric pooling for RVs sampled along x."""
    return pool_samples_f(num_f_geom_s(pdf_samples))(x)


def mult_pool_s(x: ft.NPVec, pdf_samples: Sequence[ft.DistVec]) -> ft.NPVec:
    """Performs multiplicative pooling for RVs sampled along x."""
    return pool_samples_f(num_f_mult_s(pdf_samples))(x)


def weighted_pool_s(
    x: ft.NPVec, pdf_samples: Sequence[ft.DistVec], weights: ft.NPVec
) -> ft.NPVec:
    """Performs weighted pooling for RVs sampled along x.

    Weights should add up to 1.
    """
    return pool_samples_f(num_f_weighted_s(pdf_samples, weights))(x)


"""
================================================================================================
Combined pooling convenience function using mulitple dispatch
================================================================================================
"""


@dispatch  # RV-based pooling
def pool(  # type: ignore[reportGeneralTypeIssues]
    x: ft.NPVec,
    rvs: ft.ContDists,
    weights: ft.NPVec | None = None,
    method: Literal['geom', 'mult'] = 'mult',
) -> ft.NPVec:
    """Performs pooling for both RVs and RVs sampled along x."""
    if weights is not None:
        return weighted_pool(x, rvs, weights)
    if method == 'mult':
        return mult_pool(x, rvs)
    if method == 'geom':
        return geom_pool(x, rvs)

    err_msg = 'Pooling method must be "geom" or "mult".'
    raise ValueError(err_msg)


@dispatch  # Sample-based pooling
def pool(  # noqa: F811
    x: ft.NPVec,
    rvs: Sequence[ft.DistVec],
    weights: ft.NPVec | None = None,
    method: Literal['geom', 'mult'] = 'mult',
) -> ft.NPVec:
    """Performs pooling for both RVs and RVs sampled along x."""
    if weights is not None:
        return weighted_pool_s(x, rvs, weights)
    if method == 'mult':
        return mult_pool_s(x, rvs)
    if method == 'geom':
        return geom_pool_s(x, rvs)

    err_msg = 'Pooling method must be "geom" or "mult".'
    raise ValueError(err_msg)


"""
================================================================================================
Functions with no log-sum-exp for testing purposes
================================================================================================
"""


def _pool_rvs_noexp(
    bounds: tuple[float, float],
    x: ft.NPVec,
    rvs: ft.ContDists,
    weights: ft.NPVec | None = None,
    method: Literal['geom', 'mult'] = 'mult',
) -> ft.NPVec:
    """Performs RV-based pooling without using log-sum-exp trick."""
    integrate_lo, integrate_hi = bounds
    if weights is not None:
        zipped = zip(rvs, weights, strict=True)  # for type narrowing compatibility

        def num_f(x: ft.NPVec) -> ft.NPVec:
            return np.prod([(rv.pdf(x) ** w) for rv, w in zipped], axis=0)

    elif method == 'mult':

        def num_f(x: ft.NPVec) -> ft.NPVec:
            return np.prod([rv.pdf(x) for rv in rvs], axis=0)

    elif method == 'geom':
        n_rvs = len(rvs)

        def num_f(x: ft.NPVec) -> ft.NPVec:
            return np.prod([(rv.pdf(x) ** (1 / n_rvs)) for rv in rvs], axis=0)

    else:
        err_msg = 'Pooling method must be "geom" or "mult".'
        raise ValueError(err_msg)

    return num_f(x) / quad(num_f, integrate_lo, integrate_hi)[0]


pool_rvs_noexp = partial(_pool_rvs_noexp, (0, 1))


def pool_samples_noexp(
    x: ft.NPVec,
    pdf_samples: Sequence[ft.DistVec],
    weights: ft.NPVec | None = None,
    method: Literal['geom', 'mult'] = 'mult',
) -> ft.NPVec:
    """Performs sample-based pooling without using log-sum-exp trick."""
    if weights is not None:
        num = np.prod(
            [sample**w for sample, w in zip(pdf_samples, weights, strict=True)],
            axis=0,
        )
    elif method == 'mult':
        num = np.prod(np.array(pdf_samples), axis=0)
    elif method == 'geom':
        n_rvs = len(pdf_samples)
        num = np.prod([sample ** (1 / n_rvs) for sample in pdf_samples], axis=0)
    else:
        err_msg = 'Pooling method must be "geom" or "mult".'
        raise ValueError(err_msg)

    return num / simpson(num, x)


"""
================================================================================================
Holder pooling - linear to geometric
================================================================================================
"""


def holder_pool_s(
    x: ft.NPVec,
    pdf_samples: Sequence[ft.DistVec],
    weights: ft.NPVec | None = None,
    alpha: float = 1,  # 1 is linear, limit towards 0 is geometric
) -> ft.NPVec:
    """Performs Holder pooling method using sampled RVs."""
    if weights is not None:
        num = np.sum(
            [w * sample**alpha for sample, w in zip(pdf_samples, weights, strict=True)],
            axis=0,
        ) ** (1 / alpha)
    else:
        num = np.sum([sample**alpha for sample in pdf_samples], axis=0) ** (1 / alpha)

    norm_const = simpson(num, x)
    return num / norm_const
