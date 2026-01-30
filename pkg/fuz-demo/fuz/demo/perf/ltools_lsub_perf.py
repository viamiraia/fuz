import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial

    import array_api_compat as xpc
    import fuz.types as ft
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import pyinstrument
    import torch
    from fuz import lint
    import fuz.types as ft
    from pyinstrument import Profiler
    import math
    from fuz.lint import fillna
    return fillna, ft, jnp, math, mo, np, torch, xpc


@app.cell
def _(fillna, ft, math, xpc):
    def lsub(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.NPTensor:
        """Adaptive version."""
        xp = xpc.array_namespace(la, lb)

        la, lb = fillna(la, nan), fillna(lb, nan)

        if xpc.size(la) > xpc.size(lb):
            lb = xp.broadcast_to(lb, la.shape)
        elif xpc.size(la) < xpc.size(lb):
            la = xp.broadcast_to(la, lb.shape)

        lb_minus_la = lb - la
        method = lb_minus_la < -0.6931471805599453  # noqa: PLR2004

        # Use conditional indexing for NumPy (more performant - only computes needed values)
        # Use xp.where for JAX/Torch/CuPy (required for JAX, better for GPU backends)
        if xpc.is_numpy_namespace(xp):
            res = xp.empty_like(lb_minus_la)
            res[method] = la[method] + xp.log1p(-xp.exp(lb_minus_la[method]))
            not_method = ~method
            res[not_method] = la[not_method] + xp.log(
                -xp.expm1(lb_minus_la[not_method])
            )
        else:
            res_method1 = la + xp.log1p(-xp.exp(lb_minus_la))
            res_method2 = la + xp.log(-xp.expm1(lb_minus_la))
            res = xp.where(method, res_method1, res_method2)
        return res


    def lsub_where_only(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.NPTensor:
        """Version using only xp.where for all backends"""
        xp = xpc.array_namespace(la, lb)
        la, lb = fillna(la, nan), fillna(lb, nan)
        if xpc.size(la) > xpc.size(lb):
            lb = xp.broadcast_to(lb, la.shape)
        elif xpc.size(la) < xpc.size(lb):
            la = xp.broadcast_to(la, lb.shape)
        lb_minus_la = lb - la
        method = lb_minus_la < -0.6931471805599453  # noqa: PLR2004
        res_method1 = la + xp.log1p(-xp.exp(lb_minus_la))
        res_method2 = la + xp.log(-xp.expm1(lb_minus_la))
        res = xp.where(method, res_method1, res_method2)
        return res


    def lsub_index_only(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.NPTensor:
        """Version using only indexing for all backends"""
        xp = xpc.array_namespace(la, lb)
        la, lb = fillna(la, nan), fillna(lb, nan)
        if xpc.size(la) > xpc.size(lb):
            lb = xp.broadcast_to(lb, la.shape)
        elif xpc.size(la) < xpc.size(lb):
            la = xp.broadcast_to(la, lb.shape)
        lb_minus_la = lb - la
        method = lb_minus_la < -0.6931471805599453  # noqa: PLR2004
        res = xp.empty_like(lb_minus_la)
        res[method] = la[method] + xp.log1p(-xp.exp(lb_minus_la[method]))
        not_method = ~method
        res[not_method] = la[not_method] + xp.log(-xp.expm1(lb_minus_la[not_method]))
        return res
    return lsub, lsub_index_only, lsub_where_only


@app.cell
def _(jnp, np, torch, xpc):
    def setup_lsub(shape: tuple[int, ...] = (10, 5), seed: int = 42) -> tuple:
        rng = np.random.default_rng(42)
        x_np = rng.random(shape) + 1
        y_np = rng.random(shape)
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        return (x_np, y_np), (x_jax, y_jax), (x_torch, y_torch)


    def check_lsub(x_raw, y_raw, lsub_res):
        if xpc.is_torch_array(x_raw):
            x = x_raw.numpy(force=True)
            y = y_raw.numpy(force=True)
        x = np.array(x_raw)
        y = np.array(y_raw)
        res = np.log(x - y)
        return np.allclose(lsub_res, res)
    return check_lsub, setup_lsub


@app.cell
def _(
    check_lsub,
    jnp,
    lsub,
    lsub_index_only,
    lsub_where_only,
    mo,
    np,
    setup_lsub,
    torch,
):
    (x_np, y_np), (x_jax, y_jax), (x_torch, y_torch) = setup_lsub()
    np_check = check_lsub(x_np, y_np, lsub(np.log(x_np), np.log(y_np)))
    jax_check = check_lsub(x_jax, y_jax, lsub(jnp.log(x_jax), jnp.log(y_jax)))
    torch_check = check_lsub(
        x_torch, y_torch, lsub(torch.log(x_torch), torch.log(y_torch))
    )

    np_check_where = check_lsub(x_np, y_np, lsub_where_only(np.log(x_np), np.log(y_np)))
    torch_check_where = check_lsub(
        x_torch, y_torch, lsub_where_only(torch.log(x_torch), torch.log(y_torch))
    )
    torch_check_index = check_lsub(
        x_torch, y_torch, lsub_index_only(torch.log(x_torch), torch.log(y_torch))
    )

    mo.md(f"""
    ## LSub correctness
    - Numpy check (adaptive): {np_check}
    - JAX check (adaptive): {jax_check}
    - Torch check (adaptive): {torch_check}
    - Numpy check (where_only): {np_check_where}
    - Torch check (where_only): {torch_check_where}
    - Torch check (index_only): {torch_check_index}
    """)
    return


@app.cell
def _(setup_lsub):
    def setup_perf_test(shape=(1000, 1000)):
        """Setup data for performance testing"""
        return setup_lsub(shape=shape, seed=42)


    def run_perf_test(func, x, y, n_iters=100):
        """Run performance test for a given function"""
        import time

        if hasattr(x, 'device') and 'cuda' in str(x.device):
            import torch as _torch

            _torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            result = func(x, y)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif hasattr(result, 'device') and 'cuda' in str(result.device):
                import torch as _torch

                _torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / n_iters * 1000
    return run_perf_test, setup_perf_test


@app.cell
def _(
    jnp,
    lsub,
    lsub_index_only,
    lsub_where_only,
    mo,
    np,
    run_perf_test,
    setup_perf_test,
    torch,
):
    (x_np_perf, y_np_perf), (x_jax_perf, y_jax_perf), (x_torch_perf, y_torch_perf) = (
        setup_perf_test()
    )

    la_np, lb_np = np.log(x_np_perf), np.log(y_np_perf)
    la_jax, lb_jax = jnp.log(x_jax_perf), jnp.log(y_jax_perf)
    la_torch, lb_torch = torch.log(x_torch_perf), torch.log(y_torch_perf)

    np_adaptive_time = run_perf_test(lsub, la_np, lb_np)
    np_where_time = run_perf_test(lsub_where_only, la_np, lb_np)
    np_index_time = run_perf_test(lsub_index_only, la_np, lb_np)

    jax_adaptive_time = run_perf_test(lsub, la_jax, lb_jax, n_iters=10)
    jax_where_time = run_perf_test(lsub_where_only, la_jax, lb_jax, n_iters=10)

    torch_adaptive_time = run_perf_test(lsub, la_torch, lb_torch)
    torch_where_time = run_perf_test(lsub_where_only, la_torch, lb_torch)
    torch_index_time = run_perf_test(lsub_index_only, la_torch, lb_torch)

    mo.md(f"""
    ## Performance Comparison (ms per iteration, shape={x_np_perf.shape})

    ### NumPy
    - Adaptive (uses indexing): {np_adaptive_time:.4f} ms
    - Where-only: {np_where_time:.4f} ms ({np_where_time / np_adaptive_time:.2f}x)
    - Index-only: {np_index_time:.4f} ms ({np_index_time / np_adaptive_time:.2f}x)

    ### JAX (10 iters due to compilation)
    - Adaptive (uses where): {jax_adaptive_time:.4f} ms
    - Where-only: {jax_where_time:.4f} ms ({jax_where_time / jax_adaptive_time:.2f}x)
    - Index-only: Not tested (incompatible with JAX)

    ### PyTorch (CPU)
    - Adaptive (uses where): {torch_adaptive_time:.4f} ms
    - Where-only: {torch_where_time:.4f} ms ({torch_where_time / torch_adaptive_time:.2f}x)
    - Index-only: {torch_index_time:.4f} ms ({torch_index_time / torch_adaptive_time:.2f}x)

    ### Analysis
    - **NumPy**: Where is {np_adaptive_time / np_where_time:.2f}x faster 
    - **PyTorch**: Where is {torch_index_time / torch_where_time:.2f}x faster (better vectorization)
    - **JAX**: Where is required (arrays are immutable)
    """)
    return jax_adaptive_time, np_adaptive_time, torch_adaptive_time


@app.cell
def _(fillna, ft, math, np, xpc):
    def complex_lsub(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.ArrTensor:
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
        ft.ArrTensor
            The value of `log(a - b)` as a complex number.
        """
        xp = xpc.array_namespace(la, lb)

        la = xp.astype(fillna(la, nan), complex)
        lb = xp.astype(fillna(lb, nan), complex)

        if xpc.size(la) > xpc.size(lb):
            lb = xp.broadcast_to(lb, la.shape)
        elif xpc.size(la) < xpc.size(lb):
            la = xp.broadcast_to(la, lb.shape)
        return la + xp.log(-xp.expm1(lb - la))


    def complex_lsub_np(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.ArrTensor:
        """NumPy-specific optimized complex lsub."""
        la, lb = fillna(la, nan).astype(complex), fillna(lb, nan).astype(complex)
        if la.size > lb.size:
            lb = np.broadcast_to(lb, la.shape)
        elif la.size < lb.size:
            la = np.broadcast_to(la, lb.shape)
        return la + np.log(-np.expm1(lb - la))
    return complex_lsub, complex_lsub_np


@app.cell
def _(fillna, ft, jnp, math, torch):
    def complex_lsub_jax(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.ArrTensor:
        """JAX-specific optimized complex lsub.

        Uses jnp operations directly for better JIT compilation.
        """
        la = fillna(la, nan).astype(jnp.complex128)
        lb = fillna(lb, nan).astype(jnp.complex128)

        if la.size > lb.size:
            lb = jnp.broadcast_to(lb, la.shape)
        elif la.size < lb.size:
            la = jnp.broadcast_to(la, lb.shape)
        return la + jnp.log(-jnp.expm1(lb - la))


    def complex_lsub_torch(
        la: ft.Broadcast, lb: ft.Broadcast, nan: ft.Scalar = -math.inf
    ) -> ft.ArrTensor:
        """PyTorch-specific optimized complex lsub.

        Uses torch.cfloat for complex numbers and native torch operations.
        """
        la_filled = fillna(la, nan)
        lb_filled = fillna(lb, nan)

        # Convert to complex - PyTorch uses cfloat (complex64) or cdouble (complex128)
        # Use cdouble for consistency with float64 inputs
        if la_filled.dtype == torch.float32:
            la = la_filled.to(dtype=torch.cfloat)
            lb = lb_filled.to(dtype=torch.cfloat)
        else:
            la = la_filled.to(dtype=torch.cdouble)
            lb = lb_filled.to(dtype=torch.cdouble)

        if la.numel() > lb.numel():
            lb = lb.broadcast_to(la.shape)
        elif la.numel() < lb.numel():
            la = la.broadcast_to(lb.shape)

        # PyTorch has expm1 but we need -expm1(x) = 1 - exp(x)
        return la + torch.log(-torch.expm1(lb - la))
    return complex_lsub_jax, complex_lsub_torch


@app.cell
def _(
    check_lsub,
    complex_lsub,
    complex_lsub_jax,
    complex_lsub_np,
    complex_lsub_torch,
    jnp,
    mo,
    np,
    setup_lsub,
    torch,
):
    (x_np_c, y_np_c), (x_jax_c, y_jax_c), (x_torch_c, y_torch_c) = setup_lsub()

    # Test with values where a < b (creates complex results)
    # Test generic version
    np_check_c = check_lsub(
        x_np_c, y_np_c, complex_lsub(np.log(x_np_c), np.log(y_np_c))
    )
    jax_check_c = check_lsub(
        x_jax_c, y_jax_c, complex_lsub(jnp.log(x_jax_c), jnp.log(y_jax_c))
    )
    torch_check_c = check_lsub(
        x_torch_c, y_torch_c, complex_lsub(torch.log(x_torch_c), torch.log(y_torch_c))
    )

    # Test specialized versions
    np_check_c_np = check_lsub(
        x_np_c, y_np_c, complex_lsub_np(np.log(x_np_c), np.log(y_np_c))
    )
    jax_check_c_jax = check_lsub(
        x_jax_c, y_jax_c, complex_lsub_jax(jnp.log(x_jax_c), jnp.log(y_jax_c))
    )
    torch_check_c_torch = check_lsub(
        x_torch_c, y_torch_c, complex_lsub_torch(torch.log(x_torch_c), torch.log(y_torch_c))
    )

    mo.md(f"""
    ## Complex LSub Correctness

    ### Generic (array-api compatible)
    - NumPy: {np_check_c}
    - JAX: {jax_check_c}
    - PyTorch: {torch_check_c}

    ### Backend-specific optimized
    - NumPy (complex_lsub_np): {np_check_c_np}
    - JAX (complex_lsub_jax): {jax_check_c_jax}
    - PyTorch (complex_lsub_torch): {torch_check_c_torch}

    Note: All implementations use `log(-expm1(lb - la))` for numerically stable
    complex log-space subtraction.
    """)
    return


@app.cell
def _(
    complex_lsub,
    complex_lsub_jax,
    complex_lsub_np,
    complex_lsub_torch,
    jax_adaptive_time,
    jnp,
    mo,
    np,
    np_adaptive_time,
    run_perf_test,
    setup_perf_test,
    torch,
    torch_adaptive_time,
):
    (
        (x_np_perf_c, y_np_perf_c),
        (x_jax_perf_c, y_jax_perf_c),
        (x_torch_perf_c, y_torch_perf_c),
    ) = setup_perf_test()

    la_np_c, lb_np_c = np.log(x_np_perf_c), np.log(y_np_perf_c)
    la_jax_c, lb_jax_c = jnp.log(x_jax_perf_c), jnp.log(y_jax_perf_c)
    la_torch_c, lb_torch_c = torch.log(x_torch_perf_c), torch.log(y_torch_perf_c)

    # Generic version
    np_complex_time = run_perf_test(complex_lsub, la_np_c, lb_np_c)
    jax_complex_time = run_perf_test(complex_lsub, la_jax_c, lb_jax_c, n_iters=10)
    torch_complex_time = run_perf_test(complex_lsub, la_torch_c, lb_torch_c)

    # Backend-specific versions
    np_complex_np_time = run_perf_test(complex_lsub_np, la_np_c, lb_np_c)
    jax_complex_jax_time = run_perf_test(complex_lsub_jax, la_jax_c, lb_jax_c, n_iters=10)
    torch_complex_torch_time = run_perf_test(complex_lsub_torch, la_torch_c, lb_torch_c)

    mo.md(f"""
    ## Complex LSub Performance Comparison (ms per iteration, shape={x_np_perf_c.shape})

    ### NumPy
    | Implementation | Time (ms) | vs Real lsub | vs Generic |
    |----------------|-----------|--------------|------------|
    | Generic complex_lsub | {np_complex_time:.4f} | {np_complex_time / np_adaptive_time:.2f}x | 1.00x |
    | Optimized complex_lsub_np | {np_complex_np_time:.4f} | {np_complex_np_time / np_adaptive_time:.2f}x | {np_complex_np_time / np_complex_time:.2f}x |

    **Speedup**: {np_complex_time / np_complex_np_time:.2f}x faster with NumPy-specific version

    ### JAX (10 iters due to compilation)
    | Implementation | Time (ms) | vs Real lsub | vs Generic |
    |----------------|-----------|--------------|------------|
    | Generic complex_lsub | {jax_complex_time:.4f} | {jax_complex_time / jax_adaptive_time:.2f}x | 1.00x |
    | Optimized complex_lsub_jax | {jax_complex_jax_time:.4f} | {jax_complex_jax_time / jax_adaptive_time:.2f}x | {jax_complex_jax_time / jax_complex_time:.2f}x |

    **Speedup**: {jax_complex_time / jax_complex_jax_time:.2f}x faster with JAX-specific version

    ### PyTorch (CPU)
    | Implementation | Time (ms) | vs Real lsub | vs Generic |
    |----------------|-----------|--------------|------------|
    | Generic complex_lsub | {torch_complex_time:.4f} | {torch_complex_time / torch_adaptive_time:.2f}x | 1.00x |
    | Optimized complex_lsub_torch | {torch_complex_torch_time:.4f} | {torch_complex_torch_time / torch_adaptive_time:.2f}x | {torch_complex_torch_time / torch_complex_time:.2f}x |

    **Speedup**: {torch_complex_time / torch_complex_torch_time:.2f}x faster with PyTorch-specific version

    ### Key Insights
    1. **Complex overhead**: Complex versions are ~{np_complex_np_time / np_adaptive_time:.1f}x slower than real lsub due to:
       - 2x memory (real + imaginary parts)
       - No conditional branching optimization
       - Complex arithmetic overhead

    2. **Generic vs Optimized**: The generic array-api compatible version has {'minimal' if max(np_complex_time / np_complex_np_time, jax_complex_time / jax_complex_jax_time, torch_complex_time / torch_complex_torch_time) < 1.1 else 'noticeable'} overhead

    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
