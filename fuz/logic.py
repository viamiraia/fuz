"""Perform fuzzy logic on distributions bounded between 0 and 1."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from plum import Dispatcher, dispatch, overload
from scipy.integrate import cumulative_simpson
from scipy.interpolate import BSpline, make_interp_spline

import fuz.log as flog

dispatch = Dispatcher()


def interp(x, y, lbs: tuple | None = (None), ubs: tuple | None = None):
    b = make_interp_spline(x, y)
    check_lb, check_ub = True, True
    ub = np.inf
    lb = -np.inf
    lbv = np.nan
    ubv = np.nan
    match lbs:
        case None:
            check_lb = False
        case (None, None):
            lb = x[0]
            lbv = b(lb)
        case (None, _):
            lb = x[0]
            lbv = lbs[0]
        case _:
            lb, lbv = lbs

    match ubs:
        case None:
            check_ub = False
        case (None, None):
            ub = x[-1]
            ubv = b(ub)
        case (None, _):
            ub = x[-1]
            ubv = lbs[1]
        case _:
            ub, ubv = ubs

    def f(x):
        if check_lb and x < lb:
            return lbv
        if check_ub and x > ub:
            return ubv
        return b(x)

    return f


@overload
def negf(pdf: Callable) -> Callable:
    """Get the negated distribution function bounded between 0 and 1."""
    return lambda x: pdf(1 - x)


@overload
def negf(pdf_s: np.ndarray, axis: int = 0) -> Callable:
    """negate a distribution bounded between 0 and 1 using interpolation."""
    x = np.linspace(1, 0, pdf_s.shape[axis])
    return interp(x, pdf_s, lbs=(0, 0), ubs=(1, 0))


@overload
def negf(x: np.ndarray, pdf_s: np.ndarray) -> Callable:
    """negate a distribution bounded between 0 and 1 using interpolation."""
    return interp((1 - x), pdf_s, lbs=(0, 0), ubs=(1, 0))


@dispatch
def negf() -> None:
    """Get the negated distribution function bounded between 0 and 1."""


@overload
def negs(pdf_s: np.ndarray) -> np.ndarray:
    """Get a negated distribution sample from samples between 0 and 1."""
    return np.flip(pdf_s)


@overload
def negs(x: np.ndarray, pdf: Callable) -> Callable:
    """Get a negated distribution sample."""
    return pdf(1 - x)


@overload
def negs(x: np.ndarray, pdf_s: np.ndarray) -> Callable:
    """Get a negated distribution sample using interpolation"""
    return negf(x, pdf_s)(x)


@dispatch
def negs() -> None:
    """Get the negated distribution function bounded between 0 and 1."""


# TODO(viamiraia): finish the rest of this file
@overload
def cdff(pdf: Callable, fix_order: int | None = 128, force1: bool = True) -> Callable:
    """precision is approx. log2(order)-1. So precision of 5 means order 2**6 = 64"""
    match fix_order, force1:
        case None, True:
            return lambda x: 1 if x == 1 else integrate.quad(pdf, 0, x)[0]
        case None, False:
            return lambda x: integrate.quad(pdf, 0, x)[0]
        case _, True:
            return lambda x: 1 if x == 1 else integrate.fixed_quad(pdf, 0, x, n=fix_order)[0]
        case _, _:
            return lambda x: integrate.fixed_quad(pdf, 0, x, n=fix_order)[0]


@overload
def cdff(
    x: np.ndarray, pdf_s: np.ndarray, method: Literal['trap', 'simp', 'quad']
) -> np.ndarray:
    raise NotImplementedError


@dispatch
def cdff() -> None:
    """Get the cumulative distribution function from a PDF."""


def cdfs(x, pdf_s):
    """Get the Cumulative Distribution Function (CDF) using Simpson's rule"""
    cdf = cumulative_simpson(pdf_s, x=x, initial=0)
    if math.isclose(x[-1], 1):
        cdf[-1] = 1
    return cdf


def ands(x: np.ndarray, pdf_a: np.ndarray, pdf_b: np.ndarray) -> np.ndarray:
    return pdf_a * (1 - cdfs(x, pdf_b)) + pdf_b * (1 - cdfs(x, pdf_a))


def and_pdfs(pdfs: np.ndarray, sfs: np.ndarray):
    return np.sum(pdfs * np.prod(sfs, axis=0) / sfs, axis=0)


def and_lpdfs(lpdfs: np.ndarray, lsfs: np.ndarray):
    return flog.nanlse(lpdfs + np.sum(lsfs, axis=0) - lsfs, axis=0)


def and_cdfs(cdfs: np.ndarray):
    return np.prod(cdfs, axis=0)


def and_lcdfs(lcdfs: np.ndarray):
    return np.sum(lcdfs, axis=0)


def ors(x: np.ndarray, pdf_a: np.ndarray, pdf_b: np.ndarray) -> np.ndarray:
    return pdf_a * cdfs(x, pdf_b) + pdf_b * cdfs(x, pdf_a)


def or_pdfs(pdfs: np.ndarray, cdfs: np.ndarray):
    return np.sum(pdfs * np.prod(cdfs, axis=0) / cdfs, axis=0)


def or_lpdfs(lpdfs: np.ndarray, lcdfs: np.ndarray):
    return flog.nanlse(lpdfs + np.sum(lcdfs, axis=0) - lcdfs, axis=0)


def or_sfs(sfs: np.ndarray):
    return np.prod(sfs, axis=0)


def or_lsfs(lsfs: np.ndarray):
    return np.sum(lsfs, axis=0)


def or_cdfs(cdfs: np.ndarray):
    return 1 - np.prod(1 - cdfs, axis=0)


def imps(x: np.ndarray, pdf_a: np.ndarray, pdf_b: np.ndarray) -> np.ndarray:
    """Material Implication"""
    return ors(x, negs(pdf_a), pdf_b)


def equs(x: np.ndarray, pdf_a: np.ndarray, pdf_b: np.ndarray) -> np.ndarray:
    """Material Equivalence, aka XNOR or XAND, marginally faster than equs2"""
    return ands(x, ors(x, negs(pdf_a), pdf_b), ors(x, pdf_a, negs(pdf_b)))


def equs2(x: np.ndarray, pdf_a: np.ndarray, pdf_b: np.ndarray) -> np.ndarray:
    """Material Equivalence, aka XNOR or XAND, marginally slower than equs"""
    return ors(x, ands(x, pdf_a, pdf_b), ands(x, negs(pdf_a), negs(pdf_b)))


def nands(x, pdf_a, pdf_b):
    """AKA Sheffer Stroke"""
    return negs(ands(x, pdf_a, pdf_b))


def nors(x, pdf_a, pdf_b):
    return negs(ors(x, pdf_a, pdf_b))


def xors(x, pdf_a, pdf_b):
    return ands(x, ors(x, pdf_a, pdf_b), nands(x, pdf_a, pdf_b))


def convs(x, pdf_a, pdf_b):
    """Converse is implication reversed."""
    return imps(x, pdf_b, pdf_a)
