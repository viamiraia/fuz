"""Hypothesis property-based test-suite for fuz.lint.

This module validates the numerical helpers in fuz.lint against simple
NumPy and SciPy reference implementations. Each test is powered by Hypothesis
and covers a broad range of randomly generated inputs, including edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from fuz.lint import (
    complex_lsub,
    complex_nanlse,
    fillna,
    limag_sign,
    lnorm,
    lsimp,
    lsimp13,
    lsimp38,
    lsimp_irreg,
    lsub,
    ltrap,
    nanlse,
    norm,
)
from hypothesis import assume, given, settings, strategies as st
from scipy import integrate

# ---------------------------------------------------------------------------
# Helpers & strategies
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
    """Return two 1-D NumPy arrays of equal length."""
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
    """Draw a finite 64-bit float."""
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
    return np.array(
        draw(st.lists(elements, min_size=size, max_size=size)), dtype=float
    )


@st.composite
def positive_function_data(
    draw: st.DrawFn,
    min_points: int = 3,
    max_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate x values and positive function values for integration tests."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    x = np.linspace(0.0, 1.0, n)
    # Generate positive function values
    const = draw(st.floats(min_value=1e-3, max_value=10.0))
    f = np.full_like(x, const)
    return x, f


# ---------------------------------------------------------------------------
# fillna tests
# ---------------------------------------------------------------------------


@given(
    arr=float_array(allow_nan=True),
    nan_val=finite_floats(),
)
def test_fillna(arr: np.ndarray, nan_val: float) -> None:
    """Test that fillna replaces NaN values correctly."""
    res = fillna(arr, nan_val)
    assert not np.isnan(res).any()
    nan_mask = np.isnan(arr)
    assert np.allclose(res[~nan_mask], arr[~nan_mask])
    if nan_mask.any():
        assert np.all(res[nan_mask] == nan_val)


# ---------------------------------------------------------------------------
# lsub / complex_lsub tests
# ---------------------------------------------------------------------------


@given(
    a=float_array(min_val=1e-6),
    frac=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
)
def test_lsub_matches_linear(a: np.ndarray, frac: float) -> None:
    """Test that lsub(log(a), log(b)) = log(a - b) for a > b."""
    b = a * frac
    la, lb = np.log(a), np.log(b)
    out = lsub(la, lb)
    assert np.allclose(np.exp(out), a - b, rtol=1e-9, atol=1e-9)


@given(
    a=float_array(min_val=1e-6),
    frac=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
)
def test_complex_lsub_matches_linear(a: np.ndarray, frac: float) -> None:
    """Test that complex_lsub handles subtraction correctly."""
    b = a * frac
    la, lb = np.log(a), np.log(b)
    out = complex_lsub(la, lb)
    assert np.allclose(np.exp(out).real, a - b, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# nanlse / complex_nanlse tests
# ---------------------------------------------------------------------------


@given(x=float_array(allow_nan=True, max_val=700.0))
def test_nanlse_equivalent(x: np.ndarray) -> None:
    """Test that nanlse matches manual logsumexp computation."""
    assume(~np.isnan(x).all())
    result = nanlse(x)
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
    """Test that complex_nanlse matches manual computation."""
    real, imag = data
    x = real + 1j * imag
    assume(~np.isnan(x.real).all())
    res = complex_nanlse(x)
    c = np.nanmax(x.real)
    exp_val = c + np.log(np.nansum(np.exp(x - c)))
    assert np.allclose(res, exp_val, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# limag_sign tests
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
    """Test that limag_sign returns -1 for imag close to pi, 1 otherwise."""
    real, imag = data
    x = real + 1j * imag
    res = limag_sign(x)
    assert set(np.unique(res)).issubset({-1, 1})
    near_pi = np.isclose(np.abs(x.imag), np.pi)
    assert np.all(res[near_pi] == -1)
    assert np.all(res[~near_pi] == 1)


# ---------------------------------------------------------------------------
# lnorm & norm tests
# ---------------------------------------------------------------------------


@given(x=float_array(min_val=1e-6))
def test_lnorm_normalises(x: np.ndarray) -> None:
    """Test that lnorm produces a valid probability distribution."""
    lx = lnorm(np.log(x))
    probs = np.exp(lx)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-9, atol=1e-9)


@given(x=float_array(min_val=1e-6))
def test_norm_normalises(x: np.ndarray) -> None:
    """Test that norm produces a valid probability distribution."""
    probs = norm(x)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# ltrap tests - scipy-like API
# ---------------------------------------------------------------------------


@given(data=positive_function_data())
def test_ltrap_matches_scipy_trapezoid(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that ltrap matches scipy.integrate.trapezoid."""
    x, f = data
    lfx = np.log(f)

    # Test with x array
    result = ltrap(lfx, x=x)
    expected = np.log(integrate.trapezoid(f, x=x))
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


@given(data=positive_function_data())
def test_ltrap_with_dx(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test ltrap with uniform spacing (dx parameter)."""
    x, f = data
    lfx = np.log(f)
    dx = x[1] - x[0] if len(x) > 1 else 1.0

    result = ltrap(lfx, dx=dx)
    expected = np.log(integrate.trapezoid(f, dx=dx))
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


@given(data=positive_function_data())
def test_ltrap_default_dx(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test ltrap with default dx=1.0."""
    _, f = data
    lfx = np.log(f)

    result = ltrap(lfx)
    expected = np.log(integrate.trapezoid(f))
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Known integral tests - ltrap
# ---------------------------------------------------------------------------


def test_ltrap_constant_function() -> None:
    """Integral of constant c from 0 to 1 should be c."""
    c = 5.0
    x = np.linspace(0, 1, 100)
    f = np.full_like(x, c)
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x))
    assert np.isclose(result, c, rtol=1e-6)


def test_ltrap_linear_function() -> None:
    """Integral of x from 0 to 1 should be 0.5."""
    x = np.linspace(0.01, 1, 1000)  # avoid log(0)
    f = x
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x))
    # Trapezoidal rule has some error for linear functions at boundaries
    assert np.isclose(result, 0.5, rtol=0.02)


def test_ltrap_quadratic_function() -> None:
    """Integral of x^2 from 0 to 1 should be 1/3."""
    x = np.linspace(0.01, 1, 1000)
    f = x**2
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x))
    assert np.isclose(result, 1 / 3, rtol=0.02)


def test_ltrap_exponential_function() -> None:
    """Integral of exp(x) from 0 to 1 should be e - 1."""
    x = np.linspace(0, 1, 1000)
    f = np.exp(x)
    lfx = np.log(f)  # = x

    result = np.exp(ltrap(lfx, x=x))
    expected = np.e - 1
    assert np.isclose(result, expected, rtol=1e-4)


def test_ltrap_gaussian() -> None:
    """Integral of exp(-x^2) from -5 to 5 should be close to sqrt(pi)."""
    x = np.linspace(-5, 5, 1000)
    f = np.exp(-(x**2))
    lfx = np.log(f)  # = -x^2

    result = np.exp(ltrap(lfx, x=x))
    expected = np.sqrt(np.pi)
    assert np.isclose(result, expected, rtol=1e-4)


def test_ltrap_sine_squared() -> None:
    """Integral of sin^2(x) from 0 to pi should be pi/2."""
    x = np.linspace(0.001, np.pi - 0.001, 1000)  # avoid exact zeros
    f = np.sin(x) ** 2
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x))
    expected = np.pi / 2
    assert np.isclose(result, expected, rtol=0.01)


# ---------------------------------------------------------------------------
# lsimp tests - scipy-like API
# ---------------------------------------------------------------------------


@given(data=positive_function_data(min_points=5, max_points=51))
@settings(max_examples=50)
def test_lsimp_produces_finite_result(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that lsimp produces finite results for positive functions."""
    x, f = data
    lfx = np.log(f)

    result = lsimp(lfx, x=x)
    assert np.isfinite(result)


def test_lsimp_constant_function() -> None:
    """Simpson's rule should be exact for constant functions."""
    c = 3.0
    x = np.linspace(0, 1, 101)  # odd number of points
    f = np.full_like(x, c)
    lfx = np.log(f)

    result = np.exp(lsimp(lfx, x=x))
    assert np.isclose(result, c, rtol=1e-6)


def test_lsimp_quadratic_function() -> None:
    """Simpson's rule should be exact for quadratic functions."""
    x = np.linspace(0, 1, 101)
    f = x**2 + 1  # +1 to keep positive
    lfx = np.log(f)

    result = np.exp(lsimp(lfx, x=x))
    # Integral of x^2 + 1 from 0 to 1 = 1/3 + 1 = 4/3
    expected = 4 / 3
    assert np.isclose(result, expected, rtol=1e-4)


def test_lsimp_cubic_function() -> None:
    """Simpson's 1/3 rule should be exact for cubic polynomials."""
    x = np.linspace(0, 1, 101)
    f = x**3 + 1  # +1 to keep positive
    lfx = np.log(f)

    result = np.exp(lsimp(lfx, x=x))
    # Integral of x^3 + 1 from 0 to 1 = 1/4 + 1 = 5/4
    expected = 5 / 4
    assert np.isclose(result, expected, rtol=1e-4)


def test_lsimp_with_dx() -> None:
    """Test lsimp with uniform dx parameter."""
    c = 2.0
    n = 101
    dx = 0.01
    f = np.full(n, c)
    lfx = np.log(f)

    result = np.exp(lsimp(lfx, dx=dx))
    expected = c * dx * (n - 1)  # integral over [0, 1]
    assert np.isclose(result, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# lsimp13 tests
# ---------------------------------------------------------------------------


@given(data=positive_function_data(min_points=3, max_points=51))
def test_lsimp13_positive_result(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that lsimp13 produces finite results for positive functions."""
    x, f = data
    # Ensure odd number of points for Simpson's 1/3
    if len(x) % 2 == 0:
        x = x[:-1]
        f = f[:-1]
    assume(len(x) >= 3 and len(x) % 2 == 1)

    lfx = np.log(f)
    result = lsimp13(lfx, x)
    assert np.isfinite(result)


def test_lsimp13_constant() -> None:
    """Simpson's 1/3 should be exact for constants."""
    c = 2.5
    x = np.linspace(0, 2, 11)  # odd points
    f = np.full_like(x, c)
    lfx = np.log(f)

    result = np.exp(lsimp13(lfx, x))
    expected = c * 2  # integral of c from 0 to 2
    assert np.isclose(result, expected, rtol=1e-6)


def test_lsimp13_quadratic() -> None:
    """Simpson's 1/3 should be exact for quadratic functions."""
    x = np.linspace(0, 1, 101)
    f = x**2 + 1
    lfx = np.log(f)

    result = np.exp(lsimp13(lfx, x))
    expected = 4 / 3  # integral of x^2 + 1 from 0 to 1
    assert np.isclose(result, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# lsimp38 tests
# ---------------------------------------------------------------------------


def test_lsimp38_constant() -> None:
    """Simpson's 3/8 should be exact for constants."""
    c = 2.5
    # Need n-1 intervals to be multiple of 3, so n = 3k+1
    x = np.linspace(0, 2, 10)  # 9 intervals = 3*3
    f = np.full_like(x, c)
    lfx = np.log(f)

    result = np.exp(lsimp38(lfx, x))
    expected = c * 2
    assert np.isclose(result, expected, rtol=1e-4)


def test_lsimp38_quadratic() -> None:
    """Simpson's 3/8 should be accurate for quadratic functions."""
    x = np.linspace(0, 1, 10)  # 9 intervals
    f = x**2 + 1
    lfx = np.log(f)

    result = np.exp(lsimp38(lfx, x))
    expected = 4 / 3  # integral of x^2 + 1 from 0 to 1
    assert np.isclose(result, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# lsimp_irreg tests
# ---------------------------------------------------------------------------


@given(data=positive_function_data(min_points=5))
def test_lsimp_irreg_finite(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that lsimp_irreg produces finite results."""
    x, f = data
    # Add small perturbation to make irregular
    x_irreg = x + 0.05 * np.sin(np.arange(len(x)))
    x_irreg = np.sort(x_irreg)  # ensure monotonic
    assume(np.all(np.diff(x_irreg) > 0))

    lfx = np.log(f)
    result = lsimp_irreg(lfx, x_irreg)
    assert np.isfinite(result)


def test_lsimp_irreg_constant() -> None:
    """Irregular Simpson should work for constant functions."""
    c = 3.0
    x = np.array([0, 0.3, 0.5, 0.8, 1.0])  # irregular spacing
    f = np.full_like(x, c)
    lfx = np.log(f)

    result = np.exp(lsimp_irreg(lfx, x))
    expected = c * 1.0  # integral from 0 to 1
    assert np.isclose(result, expected, rtol=0.1)


# ---------------------------------------------------------------------------
# Multi-dimensional tests
# ---------------------------------------------------------------------------


def test_ltrap_2d_axis() -> None:
    """Test ltrap along different axes for 2D arrays."""
    x = np.linspace(0, 1, 50)
    # Create 2D array: each row is a constant function
    f = np.array([[1.0] * 50, [2.0] * 50, [3.0] * 50])
    lfx = np.log(f)

    # Integrate along axis=-1 (columns)
    result = np.exp(ltrap(lfx, x=x, axis=-1))
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result, expected, rtol=1e-6)


def test_ltrap_2d_axis_0() -> None:
    """Test ltrap along axis=0."""
    x = np.linspace(0, 1, 3)
    f = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # 2x3
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x, axis=0))
    # Each column integrated: constant functions
    expected = integrate.trapezoid(f, x=x, axis=0)
    assert np.allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ltrap_minimum_points() -> None:
    """Test ltrap with minimum number of points (2)."""
    x = np.array([0.0, 1.0])
    f = np.array([1.0, 1.0])
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x))
    assert np.isclose(result, 1.0, rtol=1e-9)


def test_ltrap_with_nan() -> None:
    """Test that ltrap handles NaN values gracefully."""
    x = np.linspace(0, 1, 10)
    f = np.ones(10)
    f[5] = np.nan
    lfx = np.log(f)

    # Should still produce a result (nanlse handles NaN)
    result = ltrap(lfx, x=x)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


def test_ltrap_large_values() -> None:
    """Test ltrap with large log values (would overflow in linear space)."""
    x = np.linspace(0, 1, 100)
    # log(f) = 500, so f = exp(500) which would overflow
    lfx = np.full_like(x, 500.0)

    result = ltrap(lfx, x=x)
    # Result should be log(exp(500) * 1) = 500
    assert np.isclose(result, 500.0, rtol=1e-6)


def test_ltrap_small_values() -> None:
    """Test ltrap with small log values (would underflow in linear space)."""
    x = np.linspace(0, 1, 100)
    # log(f) = -500, so f = exp(-500) which would underflow
    lfx = np.full_like(x, -500.0)

    result = ltrap(lfx, x=x)
    # Result should be log(exp(-500) * 1) = -500
    assert np.isclose(result, -500.0, rtol=1e-6)


def test_lsimp_large_values() -> None:
    """Test lsimp with large log values."""
    x = np.linspace(0, 1, 101)
    lfx = np.full_like(x, 500.0)

    result = lsimp(lfx, x=x)
    assert np.isclose(result, 500.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Comparison with scipy for various functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'func,a,b,expected',
    [
        (lambda x: np.ones_like(x), 0, 1, 1.0),  # constant
        (lambda x: x + 1, 0, 1, 1.5),  # linear (shifted to be positive)
        (lambda x: x**2 + 1, 0, 1, 4 / 3),  # quadratic
        (lambda x: np.exp(x), 0, 1, np.e - 1),  # exponential
        (lambda x: 1 / (1 + x**2), 0, 1, np.pi / 4),  # arctan derivative
    ],
)
def test_ltrap_known_integrals(func, a, b, expected) -> None:
    """Test ltrap against known integral values."""
    x = np.linspace(a, b, 1000)
    f = func(x)
    lfx = np.log(f)

    result = np.exp(ltrap(lfx, x=x))
    assert np.isclose(result, expected, rtol=0.01)


@pytest.mark.parametrize(
    'func,a,b,expected',
    [
        (lambda x: np.ones_like(x), 0, 1, 1.0),  # constant
        (lambda x: x**2 + 1, 0, 1, 4 / 3),  # quadratic
        (lambda x: x**3 + 1, 0, 1, 5 / 4),  # cubic
    ],
)
def test_lsimp_known_integrals(func, a, b, expected) -> None:
    """Test lsimp against known integral values."""
    x = np.linspace(a, b, 101)  # odd number of points
    f = func(x)
    lfx = np.log(f)

    result = np.exp(lsimp(lfx, x=x))
    assert np.isclose(result, expected, rtol=0.01)


# ---------------------------------------------------------------------------
# Test consistency between ltrap and scipy
# ---------------------------------------------------------------------------


@given(data=positive_function_data(min_points=10, max_points=100))
@settings(max_examples=30)
def test_ltrap_scipy_consistency(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that ltrap is consistent with scipy.integrate.trapezoid."""
    x, f = data
    lfx = np.log(f)

    our_result = np.exp(ltrap(lfx, x=x))
    scipy_result = integrate.trapezoid(f, x=x)

    assert np.allclose(our_result, scipy_result, rtol=1e-9, atol=1e-12)
