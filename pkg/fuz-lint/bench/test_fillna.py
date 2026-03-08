"""Benchmarks for fillna implementations across numpy, jax, torch, and array-api-compat."""

import torch
import pytest
import numpy as np
import jax.numpy as jnp
import array_api_compat as xpc

from bench_utils import (
    SHAPES,
    DEVICES,
    make_np_array,
    make_jax_array,
    make_torch_array,
    group_name,
    run_bench,
    run_bench_mutating,
)


# ── implementations ──────────────────────────────────────────────


def fillna_np_mask_inplace(x, nan=0):
    """Fill NaN values in a NumPy array in-place via boolean mask."""
    is_nan = np.isnan(x)
    x[is_nan] = nan
    return x


def fillna_np_mask_copy(x, nan=0):
    """Fill NaN values in a NumPy array via copy + boolean mask."""
    x = x.copy()
    is_nan = np.isnan(x)
    x[is_nan] = nan
    return x


def fillna_np_where(x, nan=0):
    """Fill NaN values in a NumPy array using np.where."""
    return np.where(np.isnan(x), nan, x)


def fillna_np_n2n(x, nan=0):
    """Fill NaN values in a NumPy array using np.nan_to_num."""
    return np.nan_to_num(x, nan=nan)


def fillna_jax_where(x, nan=0):
    """Fill NaN values in a JAX array using jnp.where."""
    return jnp.where(jnp.isnan(x), nan, x)


def fillna_jax_n2n(x, nan=0):
    """Fill NaN values in a JAX array using jnp.nan_to_num."""
    return jnp.nan_to_num(x, nan=nan)


def fillna_jax_mask(x, nan=0):
    """Fill NaN values in a JAX array using .at[].set()."""
    is_nan = jnp.isnan(x)
    return x.at[is_nan].set(nan)


def fillna_torch_where(x, nan=0):
    """Fill NaN values in a PyTorch tensor using torch.where."""
    return torch.where(torch.isnan(x), nan, x)


def fillna_torch_n2n(x, nan=0):
    """Fill NaN values in a PyTorch tensor using torch.nan_to_num."""
    return torch.nan_to_num(x, nan=nan)


def fillna_torch_mask_inplace(x, nan=0):
    """Fill NaN values in a PyTorch tensor in-place via boolean mask."""
    is_nan = torch.isnan(x)
    x[is_nan] = nan
    return x


def fillna_torch_mask_copy(x, nan=0):
    """Fill NaN values in a PyTorch tensor via clone + boolean mask."""
    x = x.clone()
    is_nan = torch.isnan(x)
    x[is_nan] = nan
    return x


# xpc implementations mirror the production code path in ltools.py:fillna
# which intentionally uses different strategies per backend:
# - numpy: copy + boolean mask (numpy lacks nan_to_num in array-api namespace)
# - jax/torch: nan_to_num (functional, no mutation)


def fillna_xpc_mask(x, nan=0):
    """Fill NaN values — copy+mask."""
    xp = xpc.array_namespace(x)
    if xpc.is_numpy_array(x):
        x = x.copy()
        is_nan = xp.isnan(x)
        x[is_nan] = nan
        return x
    elif xpc.is_jax_array(x):
        is_nan = xp.isnan(x)
        return x.at[is_nan].set(nan)
    elif xpc.is_torch_array(x):
        x = x.clone()
        is_nan = xp.isnan(x)
        x[is_nan] = nan
        return x
    return xp.nan_to_num(x, nan=nan)


def fillna_xpc_where(x, nan=0):
    """Fill NaN values using where."""
    xp = xpc.array_namespace(x)
    return xp.where(xp.isnan(x), nan, x)


def fillna_xpc_n2n(x, nan=0):
    """Fill NaN values — copy+mask for numpy, nan_to_num for others."""
    xp = xpc.array_namespace(x)
    return xp.nan_to_num(x, nan=nan)


# ── numpy benchmarks ────────────────────────────────────────────


class TestFillnaNumpy:
    @pytest.mark.parametrize('shape', SHAPES)
    def test_mask_inplace(self, benchmark, shape):
        run_bench_mutating(
            benchmark,
            fillna_np_mask_inplace,
            lambda: make_np_array(shape),
            group=group_name('numpy', shape),
        )

    @pytest.mark.parametrize('shape', SHAPES)
    def test_mask_copy(self, benchmark, shape):
        x = make_np_array(shape)
        run_bench(
            benchmark, fillna_np_mask_copy, x, group=group_name('numpy', shape)
        )

    @pytest.mark.parametrize('shape', SHAPES)
    def test_where(self, benchmark, shape):
        x = make_np_array(shape)
        run_bench(benchmark, fillna_np_where, x, group=group_name('numpy', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_n2n(self, benchmark, shape):
        x = make_np_array(shape)
        run_bench(benchmark, fillna_np_n2n, x, group=group_name('numpy', shape))


# ── jax benchmarks ──────────────────────────────────────────────


class TestFillnaJax:
    @pytest.mark.parametrize('shape', SHAPES)
    def test_where(self, benchmark, shape):
        x = make_jax_array(shape)
        run_bench(benchmark, fillna_jax_where, x, group=group_name('jax', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_n2n(self, benchmark, shape):
        x = make_jax_array(shape)
        run_bench(benchmark, fillna_jax_n2n, x, group=group_name('jax', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_mask(self, benchmark, shape):
        x = make_jax_array(shape)
        run_bench(benchmark, fillna_jax_mask, x, group=group_name('jax', shape))


# ── torch benchmarks ────────────────────────────────────────────


class TestFillnaTorch:
    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_where(self, benchmark, shape, device):
        x = make_torch_array(shape, device)
        run_bench(
            benchmark,
            fillna_torch_where,
            x,
            device=device,
            group=group_name(f'torch-{device}', shape),
        )

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_n2n(self, benchmark, shape, device):
        x = make_torch_array(shape, device)
        run_bench(
            benchmark,
            fillna_torch_n2n,
            x,
            device=device,
            group=group_name(f'torch-{device}', shape),
        )

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_mask_inplace(self, benchmark, shape, device):
        run_bench_mutating(
            benchmark,
            fillna_torch_mask_inplace,
            lambda: make_torch_array(shape, device),
            device=device,
            group=group_name(f'torch-{device}', shape),
        )

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_mask_copy(self, benchmark, shape, device):
        x = make_torch_array(shape, device)
        run_bench(
            benchmark,
            fillna_torch_mask_copy,
            x,
            device=device,
            group=group_name(f'torch-{device}', shape),
        )


# ── array-api-compat benchmarks ─────────────────────────────────


class TestFillnaXpc:
    @pytest.mark.parametrize('shape', SHAPES)
    def test_mask_xpc_np(self, benchmark, shape):
        x = make_np_array(shape)
        run_bench(benchmark, fillna_xpc_mask, x, group=group_name('numpy', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_where_xpc_np(self, benchmark, shape):
        x = make_np_array(shape)
        run_bench(benchmark, fillna_xpc_where, x, group=group_name('numpy', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_n2n_xpc_np(self, benchmark, shape):
        x = make_np_array(shape)
        run_bench(benchmark, fillna_xpc_n2n, x, group=group_name('numpy', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_mask_xpc_jax(self, benchmark, shape):
        x = make_jax_array(shape)
        run_bench(benchmark, fillna_xpc_mask, x, group=group_name('jax', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_where_xpc_jax(self, benchmark, shape):
        x = make_jax_array(shape)
        run_bench(benchmark, fillna_xpc_where, x, group=group_name('jax', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_n2n_xpc_jax(self, benchmark, shape):
        x = make_jax_array(shape)
        run_bench(benchmark, fillna_xpc_n2n, x, group=group_name('jax', shape))

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_mask_xpc_torch(self, benchmark, shape, device):
        x = make_torch_array(shape, device)
        run_bench(
            benchmark,
            fillna_xpc_mask,
            x,
            device=device,
            group=group_name(f'torch-{device}', shape),
        )

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_where_xpc_torch(self, benchmark, shape, device):
        x = make_torch_array(shape, device)
        run_bench(
            benchmark,
            fillna_xpc_where,
            x,
            device=device,
            group=group_name(f'torch-{device}', shape),
        )

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('device', DEVICES)
    def test_n2n_xpc_torch(self, benchmark, shape, device):
        x = make_torch_array(shape, device)
        run_bench(
            benchmark,
            fillna_xpc_n2n,
            x,
            device=device,
            group=group_name(f'torch-{device}', shape),
        )
