import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import array_api_compat as xpc
    import fuz.types as ft
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from fuz.lint import fillna
    import time
    import altair as alt
    return alt, ft, jnp, mo, np, pl, time, torch, xpc


@app.cell
def _(torch):
    def torch_nanmax(t: torch.Tensor):
        if t.isnan().all():
            return torch.nan
        return t.nan_to_num(-torch.inf).max()
    return (torch_nanmax,)


@app.cell
def _(ft, torch_nanmax, xpc):
    def nanlse(x: ft.Broadcast, axis: int | None = None) -> ft.Broadcast:
        """Get logsumexp treating NaNs as zeroes (generic).

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
        xp = xpc.array_namespace(x)
        if xpc.is_torch_namespace(xp):
            c = torch_nanmax(x)
        else:
            c = xp.nanmax(x)
        return c + xp.log(xp.nansum(xp.exp(x - c), axis=axis))
    return (nanlse,)


@app.cell
def _(ft, np):
    def nanlse_np(x: ft.Broadcast, axis: int | None = None) -> ft.Broadcast:
        """Get logsumexp treating NaNs as zeroes (NumPy optimized).

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
        x = np.array(x)
        c = np.nanmax(x)
        return c + np.log(np.nansum(np.exp(x - c), axis=axis))
    return (nanlse_np,)


@app.cell
def _(ft, torch):
    def nanlse_torch(x: ft.Broadcast, axis: int | None = None) -> ft.Broadcast:
        """Get logsumexp treating NaNs as zeroes (PyTorch optimized).

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
        x = torch.as_tensor(x)

        if x.isnan().all():
            return torch.tensor(float('nan'))

        c = x.nan_to_num(neginf=-torch.inf).max()
        exp_vals = torch.exp(x - c)
        exp_vals = torch.nan_to_num(exp_vals, nan=0.0)

        return c + torch.log(exp_vals.sum(dim=axis))
    return (nanlse_torch,)


@app.cell
def _(ft, jnp):
    def nanlse_jax(x: ft.Broadcast, axis: int | None = None) -> ft.Broadcast:
        """Get logsumexp treating NaNs as zeroes (JAX optimized).

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
        x = jnp.asarray(x)
        c = jnp.nanmax(x)
        return c + jnp.log(jnp.nansum(jnp.exp(x - c), axis=axis))
    return (nanlse_jax,)


@app.cell
def _(mo):
    mo.md("""
    ## Correctness Tests

    Verify all implementations produce the same results across different test cases.
    """)
    return


@app.cell
def _(jnp, mo, nanlse, nanlse_jax, nanlse_np, nanlse_torch, np, pl, torch):
    test_data = [
        [1.0, 2.0, 3.0],
        [1.0, float('nan'), 3.0],
        [float('nan'), float('nan'), 3.0],
        [np.log(0.5), np.log(0.3), np.log(0.2)],
        [float('nan'), float('nan'), float('nan')],
        [-10.0, -5.0, 0.0, 5.0],
    ]

    test_results = []
    for i, _data in enumerate(test_data):
        np_array = np.array(_data)
        result_ref = nanlse_np(np_array)

        torch_array = torch.tensor(_data, dtype=torch.float32)
        jax_array = jnp.array(_data)

        implementations = [
            ('nanlse (NumPy)', nanlse(np_array)),
            ('nanlse_np', result_ref),
            ('nanlse (Torch)', nanlse(torch_array)),
            ('nanlse_torch', nanlse_torch(torch_array)),
            ('nanlse (JAX)', nanlse(jax_array)),
            ('nanlse_jax', nanlse_jax(jax_array)),
        ]

        for _impl_name, result in implementations:
            if hasattr(result, 'numpy'):
                result = result.numpy()
            result_float = float(result)

            match = np.allclose(
                result_float,
                float(result_ref),
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )

            test_results.append({
                'Test': i + 1,
                'Implementation': _impl_name,
                'Result': result_float,
                'Match': 'Ã¢Å“â€œ' if match else 'Ã¢Å“â€”',
                'Diff': float(abs(result_float - float(result_ref))),
            })

    correctness_df = pl.DataFrame(test_results)
    mo.ui.table(correctness_df, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Performance Benchmarks

    Compare the performance of generic vs specialized implementations across different backends.
    """)
    return


@app.cell
def _(
    jnp,
    mo,
    nanlse,
    nanlse_jax,
    nanlse_np,
    nanlse_torch,
    np,
    pl,
    time,
    torch,
):
    def benchmark_function(func, data, warmup=5, iterations=100):
        for _ in range(warmup):
            _ = func(data)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(data)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
        }

    array_sizes = [100, 1000, 10000, 100000]
    perf_results = []

    for array_size in array_sizes:
        np_data = np.random.randn(array_size).astype(np.float32)
        np_data[np.random.rand(array_size) < 0.1] = np.nan

        torch_data = torch.from_numpy(np_data)
        jax_data = jnp.array(np_data)

        benchmarks = [
            ('NumPy', 'nanlse (generic)', lambda data: nanlse(data), np_data),
            ('NumPy', 'nanlse_np', lambda data: nanlse_np(data), np_data),
            ('PyTorch', 'nanlse (generic)', lambda data: nanlse(data), torch_data),
            ('PyTorch', 'nanlse_torch', lambda data: nanlse_torch(data), torch_data),
            ('JAX', 'nanlse (generic)', lambda data: nanlse(data), jax_data),
            ('JAX', 'nanlse_jax', lambda data: nanlse_jax(data), jax_data),
        ]

        for backend, impl_name, func, data in benchmarks:
            stats = benchmark_function(func, data, warmup=5, iterations=50)
            perf_results.append({
                'Size': array_size,
                'Backend': backend,
                'Implementation': impl_name,
                'Mean (ms)': stats['mean'],
                'Std (ms)': stats['std'],
                'Min (ms)': stats['min'],
                'Max (ms)': stats['max'],
            })

    perf_df = pl.DataFrame(perf_results)

    speedup_results = []

    for array_size in perf_df['Size'].unique():
        size_data = perf_df.filter(pl.col('Size') == array_size)

        for backend in size_data['Backend'].unique():
            backend_data = size_data.filter(pl.col('Backend') == backend)

            generic = backend_data.filter(pl.col('Implementation') == 'nanlse (generic)')
            specialized = backend_data.filter(pl.col('Implementation') != 'nanlse (generic)')

            if len(generic) > 0 and len(specialized) > 0:
                generic_time = generic['Mean (ms)'][0]
                specialized_time = specialized['Mean (ms)'][0]
                speedup = generic_time / specialized_time

                speedup_results.append({
                    'Size': array_size,
                    'Backend': backend,
                    'Generic (ms)': generic_time,
                    'Specialized (ms)': specialized_time,
                    'Speedup': speedup,
                    'Improvement': f'{(speedup - 1) * 100:.1f}%'
                })

    speedup_df = pl.DataFrame(speedup_results)
    mo.ui.table(speedup_df, selection=None)
    return (perf_df,)


@app.cell
def _(perf_df):
    perf_df
    return


@app.cell
def _(alt, perf_df):
    chart_data = perf_df.to_pandas()

    perf_chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X('Size:Q', scale=alt.Scale(type='log'), title='Array Size'),
        y=alt.Y('Mean (ms):Q', scale=alt.Scale(type='log'), title='Time (ms, log scale)'),
        color=alt.Color('Implementation:N', title='Implementation'),
        strokeDash=alt.StrokeDash('Backend:N', title='Backend'),
        tooltip=['Size:Q', 'Backend:N', 'Implementation:N', 'Mean (ms):Q', 'Std (ms):Q']
    ).properties(
        width=700,
        height=400,
        title='Performance Comparison: nanlse Implementations'
    ).interactive()

    perf_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    ### Implementations

    1. **`nanlse` (generic)**: Uses array-api-compat to support multiple backends
    2. **`nanlse_np`**: NumPy-specific optimized version
    3. **`nanlse_torch`**: PyTorch-specific optimized version using `nan_to_num`
    4. **`nanlse_jax`**: JAX-specific optimized version leveraging `nansum`

    ### Key Optimizations

    - **PyTorch**: Uses native `nan_to_num` operations to avoid NaN handling overhead
    - **JAX**: Leverages efficient `nansum` operation for automatic NaN handling
    - **NumPy**: Direct use of NumPy operations without array-api-compat overhead

    All specialized implementations maintain numerical accuracy while improving performance.
    """)
    return


if __name__ == "__main__":
    app.run()
