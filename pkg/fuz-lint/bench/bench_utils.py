"""Shared benchmark infrastructure for fuz-lint benchmarks."""

import torch
import numpy as np
import jax.numpy as jnp

SEED = 42
NAN_PROB = 0.75
SHAPES = [(100,), (10_000, 10), (1_000, 1_000)]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def make_np_array(shape, seed=SEED, nan_prob=NAN_PROB):
    rng = np.random.default_rng(seed)
    x = rng.random(shape)
    x[rng.random(shape) < nan_prob] = np.nan
    return x


def make_jax_array(shape, seed=SEED, nan_prob=NAN_PROB):
    return jnp.array(make_np_array(shape, seed, nan_prob))


def make_torch_array(shape, device="cpu", seed=SEED, nan_prob=NAN_PROB):
    return torch.from_numpy(make_np_array(shape, seed, nan_prob)).to(device)


def group_name(framework, shape):
    dims = "x".join(str(d) for d in shape)
    return f"{framework} ({dims})"


def _cuda_sync(fn, x, **kwargs):
    torch.cuda.synchronize()
    result = fn(x, **kwargs)
    torch.cuda.synchronize()
    return result


def run_bench(benchmark, fn, x, *, device="cpu", group=None):
    """Run benchmark, handling CUDA sync and group assignment."""
    if group is not None:
        benchmark.group = group
    if device == "cuda":
        benchmark(_cuda_sync, fn, x)
    else:
        benchmark(fn, x)


def run_bench_mutating(benchmark, fn, make_array, *, device="cpu", group=None,
                       rounds=100, iterations=1, warmup_rounds=5):
    """Run benchmark for functions that mutate their input.

    Uses benchmark.pedantic with a setup function to create a fresh array
    for every iteration, preventing corruption across runs.
    """
    if group is not None:
        benchmark.group = group

    if device == "cuda":
        def target(fn, x):
            torch.cuda.synchronize()
            result = fn(x)
            torch.cuda.synchronize()
            return result
    else:
        def target(fn, x):
            return fn(x)

    def setup():
        return (fn, make_array()), {}

    benchmark.pedantic(target, setup=setup, rounds=rounds,
                       iterations=iterations, warmup_rounds=warmup_rounds)
