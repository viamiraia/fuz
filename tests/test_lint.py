"""
Hypothesis property-based test-suite for :pymod:`fuz.log`.

The module validates the numerical helpers in :pymod:`fuz.log` against simple
NumPy reference implementations.  Each test is powered by Hypothesis and thus
covers a broad range of randomly generated inputs, including edge cases such as
arrays containing ``NaN`` values or extreme magnitudes.

Notes
-----
The suite is intentionally defensive: it constrains the random inputs to avoid
undefined behaviour (e.g. subtracting a larger number from a smaller one in
log-space) and relaxes tolerances where the target algorithm is only an
approximation (e.g. Simpson integration rules).

The goal is to assert key mathematical identities for each public helper in
``fuz.log`` across a wide range of inputs.  The tests intentionally avoid edge
cases that are explicitly undefined (e.g. subtraction where *b > a* for
``lsub``) and focus instead on correctness of the implemented formulae.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st
from hypothesis.extra import numpy as hnp

from fuz.lint import (
    complex_lsub,
    complex_nanlse,
    fillna,
    limag_sign,
    lnorm,
    log_trap,
    lsimp13,
    lsimp38,
    lsimp_irreg,
    lsub,
    ltrap,
    nanlse,
    norm,
)

# ---------------------------------------------------------------------------
# Helpers & strategies
# ---------------------------------------------------------------------------


Shape = st.integers(min_value=1, max_value=10)


# ---------------------------------------------------------------------------
# Paired array helper
# ---------------------------------------------------------------------------


@st.composite
def paired_float_arrays(
    draw: st.DrawFn,
    *,
    min_val_a: float = -1.0,
    max_val_a: float = 1.0,
    min_val_b: float = -1.0,
    max_val_b: float = 1.0,
    size_bounds: tuple[int, int] = (1, 10),
) -> tuple[np.ndarray, np.ndarray]:
    """Return two 1-D NumPy arrays of **equal length**.

    Helps avoid Hypothesis health-check failures from filtering when we need
    arrays with matching shapes.
    """
    size = draw(st.integers(*size_bounds))
    arr_a = draw(
        float_array(
            min_size=size,
            max_size=size,
            min_val=min_val_a,
            max_val=max_val_a,
            allow_nan=False,
        )
    )
    arr_b = draw(
        float_array(
            min_size=size,
            max_size=size,
            min_val=min_val_b,
            max_val=max_val_b,
            allow_nan=False,
        )
    )
    return arr_a, arr_b


@st.composite
def finite_floats(draw: st.DrawFn, *, allow_nan: bool = False) -> float:
    """Draw a finite 64-bit float.

    Hypothesis forbids combining ``allow_nan=True`` with numeric bounds.  We
    therefore choose bounds *only* when NaNs are disallowed.
    """
    if allow_nan:
        strat = st.floats(allow_nan=True, allow_infinity=False, width=64)
    else:
        strat = st.floats(
            min_value=-1.0e6,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        )
    return draw(strat)


@st.composite
def float_array(
    draw: st.DrawFn,
    *,
    allow_nan: bool = False,
    min_size: int = 1,
    max_size: int = 10,
    min_val: float | None = -1e6,
    max_val: float | None = 1e6,
) -> np.ndarray:
    """Return a 1-D NumPy array of floats."""
    size = draw(st.integers(min_size, max_size))
    if allow_nan:
        elements = st.floats(allow_nan=True, allow_infinity=False, width=64)
    else:
        elements = st.floats(
            min_value=min_val,
            max_value=max_val,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        )
    return np.array(draw(st.lists(elements, min_size=size, max_size=size)), dtype=float)


# ---------------------------------------------------------------------------
# fillna
# ---------------------------------------------------------------------------


@given(
    arr=float_array(allow_nan=True),
    nan_val=finite_floats(),
)
def test_fillna(arr: np.ndarray, nan_val: float) -> None:
    res = fillna(arr, nan_val)
    assert not np.isnan(res).any()

    nan_mask = np.isnan(arr)
    # Non-NaNs unchanged
    assert np.allclose(res[~nan_mask], arr[~nan_mask])
    # NaNs replaced
    if nan_mask.any():
        assert np.all(res[nan_mask] == nan_val)


# ---------------------------------------------------------------------------
# lsub / complex_lsub
# ---------------------------------------------------------------------------


@given(
    a=float_array(min_val=1e-6),
    frac=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
)
def test_lsub_matches_linear(a: np.ndarray, frac: float) -> None:
    b = a * frac  # ensures b <= a element-wise
    la, lb = np.log(a), np.log(b)
    out = lsub(la, lb)
    assert np.allclose(np.exp(out), a - b, rtol=1e-9, atol=1e-9)


@given(
    a=float_array(min_val=1e-6),
    frac=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
)
def test_complex_lsub_matches_linear(a: np.ndarray, frac: float) -> None:
    b = a * frac
    la, lb = np.log(a), np.log(b)
    out = complex_lsub(la, lb)
    assert np.allclose(np.exp(out).real, a - b, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# nanlse / complex_nanlse
# ---------------------------------------------------------------------------


@given(x=float_array(allow_nan=True, max_val=700.0))
def test_nanlse_equivalent(x: np.ndarray) -> None:
    assume(~np.isnan(x).all())
    result = nanlse(x)
    # replicate stable computation
    c = np.nanmax(x)
    expected = c + np.log(np.nansum(np.exp(x - c)))
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


@given(
    data=paired_float_arrays(
        min_val_a=-100.0,
        max_val_a=100.0,
        min_val_b=-10.0,
        max_val_b=10.0,
    )
)
def test_complex_nanlse_equivalent(data: tuple[np.ndarray, np.ndarray]) -> None:
    real, imag = data
    x = real + 1j * imag
    assume(~np.isnan(x.real).all())

    res = complex_nanlse(x)
    c = np.nanmax(x.real)
    exp_val = c + np.log(np.nansum(np.exp(x - c)))
    assert np.allclose(res, exp_val, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# limag_sign
# ---------------------------------------------------------------------------


@given(
    data=paired_float_arrays(
        min_val_a=-10.0,
        max_val_a=10.0,
        min_val_b=-4 * np.pi,
        max_val_b=4 * np.pi,
    )
)
def test_limag_sign(data: tuple[np.ndarray, np.ndarray]) -> None:
    real, imag = data
    x = real + 1j * imag
    res = limag_sign(x)
    assert set(np.unique(res)).issubset({-1, 1})

    near_pi = np.isclose(np.abs(x.imag), np.pi)
    assert np.all(res[near_pi] == -1)
    assert np.all(res[~near_pi] == 1)


# ---------------------------------------------------------------------------
# lnorm & norm
# ---------------------------------------------------------------------------


@given(x=float_array(min_val=1e-6))
def test_lnorm_normalises(x: np.ndarray) -> None:
    lx = lnorm(np.log(x))
    probs = np.exp(lx)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-9, atol=1e-9)


@given(x=float_array(min_val=1e-6))
def test_norm_normalises(x: np.ndarray) -> None:
    probs = norm(x)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Integration rules
# ---------------------------------------------------------------------------


@st.composite
def positive_fun(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=3, max_value=30))
    x = np.linspace(0.0, 1.0, n)
    const = draw(st.floats(min_value=1e-3, max_value=10.0))
    f = np.full_like(x, const)
    return x, f


@given(data=positive_fun())
def test_log_trap_matches_numpy(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, f = data
    fx = np.log(f)
    res = log_trap(fx, x)
    expected = np.log(np.trapz(f, x))
    assert np.allclose(res, expected, rtol=1e-9, atol=1e-9)


@given(data=positive_fun())
def test_ltrap_dx_and_x(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, f = data
    fx = np.log(f)
    dx = x[1] - x[0]

    # variant accepting dx
    res_dx = ltrap(fx, dx)
    # variant accepting x vector
    res_x = ltrap(fx, x)
    expected = np.log(np.trapz(f, x))
    assert np.allclose(res_dx, expected, rtol=1e-9, atol=1e-9)
    assert np.allclose(res_x, expected, rtol=1e-9, atol=1e-9)


@given(data=positive_fun())
def test_simpson_rules(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, f = data
    fx = np.log(f)
    # For Simpson rules require length conditions; guard here
    if (x.size - 1) % 2 == 0:
        assert np.allclose(np.exp(lsimp13(fx, x)), np.trapz(f, x), rtol=1e-6, atol=1e-6)
    if (x.size - 1) % 3 == 0:
        assert np.allclose(np.exp(lsimp38(fx, x)), np.trapz(f, x), rtol=1e-6, atol=1e-6)

    # Irregular spacing: perturb x slightly while keeping order
    x_irreg = x + 0.1 * (x - x.mean())
    assert np.all(np.diff(x_irreg) > 0)
    fx_irreg = np.log(f)
    assert np.allclose(
        np.exp(lsimp_irreg(fx_irreg, x_irreg)), np.trapz(f, x_irreg), rtol=1e-6, atol=1e-6
    )
