import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    - `fillna` is good enough. The overhead of `array_api_compat` seems to be a one-time thing. It's not too much so it is worth using.
    """)
    return


@app.cell
def _(setup):
    xs = []
    for i in range(50):
        _x = setup((10_000,1_000))
        xs.append(_x)
    return (xs,)


@app.cell
def _(fillna, fillna3):
    # just to distinguish the names during profiling
    def fillna_np_wrapper(x):
        return fillna(x)

    def fillna_jax_wrapper(x):
        return fillna(x)

    def fillna_torch_wrapper(x):
        return fillna(x)

    def fillna_np_wrapper3(x):
        return fillna3(x)

    def fillna_jax_wrapper3(x):
        return fillna3(x)

    def fillna_torch_wrapper3(x):
        return fillna3(x)

    return (
        fillna_jax_wrapper,
        fillna_jax_wrapper3,
        fillna_np_wrapper,
        fillna_np_wrapper3,
        fillna_torch_wrapper,
        fillna_torch_wrapper3,
    )


@app.cell
def _(
    Profiler,
    fillna_jax,
    fillna_jax2,
    fillna_jax_wrapper,
    fillna_jax_wrapper3,
    fillna_np,
    fillna_np2,
    fillna_np_wrapper,
    fillna_np_wrapper3,
    fillna_torch,
    fillna_torch2,
    fillna_torch3,
    fillna_torch_wrapper,
    fillna_torch_wrapper3,
    mo,
    xs,
):
    with Profiler(interval=0.0001) as _p:
        for (_np, _jax, _torch) in xs:
            fillna_np2(_np)
            fillna_np(_np)
            fillna_np_wrapper(_np)
            fillna_np_wrapper3(_np)
            fillna_jax(_jax)
            fillna_jax2(_jax)
            fillna_jax_wrapper(_jax)
            fillna_jax_wrapper3(_jax)
            fillna_torch(_torch)
            fillna_torch2(_torch)
            fillna_torch3(_torch)
            fillna_torch_wrapper(_torch)
            fillna_torch_wrapper3(_torch)

    mo.iframe(_p.output_html())
    return


@app.cell(column=1)
def _():
    import marimo as mo
    import fuz.types as ft
    import numpy as np
    import jax.numpy as jnp
    import torch
    import array_api_compat as xpc
    import pyinstrument
    from pyinstrument import Profiler
    from fuz import lint
    from functools import partial
    from typing import TypeAlias
    from attrs import frozen
    import attrs

    return Profiler, TypeAlias, frozen, jnp, mo, np, torch, xpc


@app.cell
def _(TypeAlias, frozen, torch):
    DevType: TypeAlias = str | torch.device

    @frozen
    class TorchCfg:
        device: DevType
        seed: int | None = None

    def setup_torch(seed: int | None = None) -> TorchCfg:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(device)
        tcfg = TorchCfg(device=device, seed=seed)
        return tcfg


    def handle_tcfg(tcfg: TorchCfg | None = None):
        if tcfg is None:
            return torch.get_default_device()
        device = torch.get_default_device() if tcfg.device is None else tcfg.device
        if tcfg.seed is not None and torch.seed() != tcfg.seed:
            torch.manual_seed(tcfg.seed)
        return device


    tcfg = setup_torch(seed=42)
    tcfg
    return


@app.cell
def _(torch):
    for _i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(_i).name)
    return


@app.cell
def _(jnp, np, torch):
    def fillna_np(x, nan=0):
        """Fill NaN values in a NumPy array with a specified value."""
        is_nan = np.isnan(x)
        x[is_nan] = nan
        return x

    def fillna_np2(x, nan=0):
        """Fill NaN values in a NumPy array using np.where."""
        return np.where(np.isnan(x), nan, x)

    def fillna_jax(x, nan=0):
        """Fill NaN values in a JAX array using jnp.where."""
        return jnp.where(jnp.isnan(x), nan, x)

    def fillna_jax2(x, nan=0):
        """Fill NaN values in a JAX array using jnp.nan_to_num."""
        return jnp.nan_to_num(x, nan=nan)

    def fillna_torch(x, nan=0):
        """Fill NaN values in a PyTorch tensor using torch.where."""
        return torch.where(torch.isnan(x), nan, x)

    def fillna_torch2(x, nan=0):
        """Fill NaN values in a PyTorch tensor using torch.nan_to_num."""
        return torch.nan_to_num(x, nan=nan)

    def fillna_torch3(x, nan=0):
        """Fill NaN values in a PyTorch tensor"""
        is_nan = torch.isnan(x)
        x[is_nan] = nan
        return x

    return (
        fillna_jax,
        fillna_jax2,
        fillna_np,
        fillna_np2,
        fillna_torch,
        fillna_torch2,
        fillna_torch3,
    )


@app.cell
def _(jnp, np, torch, xpc):
    def fillna(x, nan = 0):
        """Fill NaN values with specified value."""
        xp = xpc.array_namespace(x)
        if xpc.is_numpy_array(x):
            is_nan = xp.isnan(x)
            x[is_nan] = nan
            return x
        return xp.nan_to_num(x, nan=nan)

    def fillna2(x, nan = 0):
        """Fill NaN values using where."""
        xp = xpc.array_namespace(x)
        return xp.where(xp.isnan(x), nan, x)

    def fillna3(x, nan = 0):
        """Fill NaN values with array-specific method."""
        if xpc.is_numpy_array(x):
            is_nan = np.isnan(x)
            x[is_nan] = nan
            return x
        elif xpc.is_jax_array(x):
            return jnp.nan_to_num(x, nan=nan)
        elif xpc.is_torch_array(x):
            return torch.nan_to_num(x, nan=nan)
        xp = xpc.array_namespace(x)   
        return xp.nan_to_num(x, nan=nan)

    return fillna, fillna3


@app.cell
def _(jnp, np, torch):
    def setup(shape: tuple[int, ...] = (10, 5), nan_prob: float = 0.75, seed: int = 42, pt_device: str = 'cuda') -> tuple[np.ndarray, jnp.ndarray, torch.Tensor]:
        rng = np.random.default_rng(42)
        x_np = rng.random(shape)
        mask = rng.random(shape) < nan_prob
        x_np[mask] = np.nan
        x_jax = jnp.array(x_np)
        x_torch = torch.from_numpy(x_np).to(pt_device)
        return x_np, x_jax, x_torch

    return (setup,)


@app.cell
def _(fillna, np):
    _x = np.empty(10)
    _x[:] = np.nan
    fillna(_x)
    _x
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
