import marimo

__generated_with = "0.18.1"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    <h1 align='center'>fuz-lint</h1>
    <h3 align='center'>chapter 1: integration in log space</h3>
    <p align='center'>by miraia s. chiou © 2025</p>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    doing numerical integration in logarithmic space allows one to be significantly more accurate when it comes to very small or very large numbers, which tend to push the limits of floating-point accuracy. `fuz-lint` provides a log result that can be more precisely manipulated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## benefits

    ### numerical stability

    When dealing with probabilities, especially in high-dimensional spaces or long sequences (like in time-series analysis), values can become infinitesimally small. Standard floating-point arithmetic would round these to zero (underflow). Working in log-space transforms these small numbers into manageable negative numbers (e.g., $\ln(10^{-100}) \approx -230$), preserving precision.

    ### dynamic range

    allows for the integration of functions that span many orders of magnitude, which would otherwise be difficult to represent and integrate accurately using linear scales.

    ### accuracy for specific functions

    functions that are "smooth" in log-space (like power laws or exponentials) can be integrated more accurately with fewer sample points.

    ## applications

    - bayesian inference
    - physics & astronomy
    - statistical mechanics
    - machine learning
    """)
    return


@app.cell
def _(Beta, np):
    scale = 10
    b1 = Beta(2, np.round(np.exp(scale)) + 1)
    lx1 = np.linspace(-scale-5,0,10000)
    x1 = np.exp(lx1)
    y1 = b1.pdf(x1)
    ly1 = b1.logpdf(x1)
    def b1_lpdf(x, args):
        return b1.logpdf(x)
    return b1_lpdf, lx1, ly1, x1, y1


@app.cell
def _(lint, np):
    def ltrap2(fx: np.ndarray, x: np.ndarray) -> float:
        """Log-space trapezoidal integration for non-uniformly spaced data.

        This is a specialized version of :func:`log_trap` for non-uniformly spaced
        data.

        Parameters
        ----------
        ly
            Logarithm of function values to integrate.
        lx
            Logarithm of the x axis.

        Returns
        -------
        float
            The logarithm of the definite integral.
        """
        return lint.nanlse(
            lint.complex_lsub(x[1:],x[:-1]) - np.log(2) + np.logaddexp(fx[:-1], fx[1:])
        )
    def ltrap3(fx: np.ndarray, x: np.ndarray) -> float:
        return lint.nanlse(
            lint.lsub(x[1:],x[:-1]) - np.log(2) + np.logaddexp(fx[:-1], fx[1:])
        )
    return (ltrap3,)


@app.cell
def _():
    return


@app.cell
def _(ltrap3, lx1, ly1):

    ltrap3(ly1,lx1)
    return


@app.cell
def _(trapezoid, x1, y1):
    trapezoid(y1, x1)
    return


@app.cell
def _(b1_lpdf, lint, logtrapz, ly1, np, trapezoid, x1, y1):
    lint.ltrap(ly1, x1), np.log(trapezoid(y1, x1)), logtrapz(b1_lpdf,x1)
    return


@app.cell
def _(np, simpson, x1, y1):
    np.log(simpson(y1, x1))
    return


@app.cell
def _(lint, ly1, x1):
    lint.lsimp_irreg(ly1,x1)
    return


@app.cell
def _():
    return


@app.cell
def _(beta1, np, pl):


    # Create a range of x values in [0, 1] for the Beta distribution
    xax = np.linspace(0, 1, 200)
    pdf_vals = beta1.pdf(xax)  # assume Beta object has a pdf method

    beta_df = pl.DataFrame({"x": xax, "density": pdf_vals})
    return (beta_df,)


@app.cell
def _(alt, beta_df):
    # Altair chart of the Beta PDF with tooltips and interactivity
    beta_chart = (
        alt.Chart(beta_df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("density:Q", title="Density"),
            tooltip=["x:Q", "density:Q"],
        )
        .interactive()
    )

    beta_chart
    return


@app.cell
def _(np):
    def expf(x):
        return np.exp(x) * np.sqrt(1 + np.exp(x))

    def expf_int(x): 
        """Get the analytical indefinite integral of expf1"""
        return (2/3) * np.pow(1+np.exp(x), 1.5)

    def expf_def_int(x1, x2):
        """Get the analytical definite integral of expf1 from x1 to x2."""
        return expf_int(x2) - expf_int(x1)

    def lexpf(x):
        """Log of expf1."""
        return x + 0.5*np.log1p(np.exp(x)) 
    return expf, expf_def_int, lexpf


@app.cell
def _():
    return


@app.cell
def _(expf, expf_def_int, lexpf, lint, np, trapezoid):
    expf_lb1, expf_ub1 = 50, 55

    expf_dx = 0.001
    expf_x1 = np.arange(expf_lb1, expf_ub1, expf_dx)
    expf_y1 = expf(expf_x1)
    res_sp1 = trapezoid(expf_y1, expf_x1)
    res_an1 = expf_def_int(expf_lb1, expf_ub1)

    lexpf_y1 = lexpf(expf_x1)
    res_lint1 = lint.ltrap(lexpf_y1, expf_dx)

    expf_lb2, expf_ub2 = 55, 60
    expf_x2 = np.arange(expf_lb2, expf_ub2, expf_dx)
    expf_y2 = expf(expf_x2)
    res_sp2 = trapezoid(expf_y2, expf_x2)
    res_an2 = expf_def_int(expf_lb2, expf_ub2)

    lexpf_y2 = lexpf(expf_x2)
    res_lint2 = lint.ltrap(lexpf_y2, expf_dx)

    (
    np.log(res_sp1) + np.log(res_sp2),
    np.log(res_an1) + np.log(res_an2),
    res_lint1 + res_lint2
    )
    return


@app.cell
def _():
    return


@app.cell
def _(lcquad, logtrapz, lqag, lqng, np):
    # define the log of the function to be integrated
    def _integrand(x, args):
        mu, sig = args # unpack extra arguments
        return -0.5*((x-mu)/sig)**2

    # set integration limits
    xmin = -6.
    xmax = 6.

    # set additional arguments
    mu = 0.
    sig = 1.

    resqag = lqag(_integrand, xmin, xmax, args=(mu, sig))
    resqng = lqng(_integrand, xmin, xmax, args=(mu, sig))
    rescquad = lcquad(_integrand, xmin, xmax, args=(mu, sig))
    restrapz = logtrapz(_integrand, np.linspace(xmin, xmax, 100), args=(mu, sig))
    restrapz
    return


@app.cell
def _():
    return


@app.cell
def _(expf1_x, lexpf1_2, logtrapz):
    logtrapz(lexpf1_2, expf1_x)
    return


@app.cell
def _(expf1_dx, lexpf1_y, lint, np):

    np.exp(lint.ltrap(lexpf1_y, expf1_dx))
    return


@app.cell
def _(expf1_dx, lexpf1_y, lint):
    lint.ltrap(lexpf1_y, expf1_dx)
    return


@app.cell
def _(np, pl):

    # Define the range for x
    x_vals = np.linspace(-4, 3, 500)  # choose a range that keeps e^x reasonable

    # Original integrand: f(x) = e^x * sqrt(1 + e^x)
    exp_x = np.exp(x_vals)
    integrand = exp_x * np.sqrt(1 + exp_x)

    # Analytic antiderivative:
    # Let u = 1 + e^x => du = e^x dx
    # ∫ e^x √(1+e^x) dx = ∫ √u du = (2/3) u^{3/2} + C
    antiderivative = (2/3) * np.power(1 + exp_x, 1.5)

    # Create a Polars DataFrame for Altair
    df_int = pl.DataFrame({
        "x": x_vals,
        "integrand": integrand,
        "antiderivative": antiderivative,
    })
    return (df_int,)


@app.cell
def _(alt, df_int):
    # Altair chart showing both the integrand and its analytic integral
    int_chart = (
        alt.Chart(df_int)
        .transform_fold(
            ["integrand", "antiderivative"],
            as_=["type", "y"]
        )
        .mark_line()
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("y:Q", title="Value"),
            color=alt.Color("type:N", legend=alt.Legend(title="Curve")),
            tooltip=["type:N", "x:Q", "y:Q"]
        )
        .properties(
            width=600,
            height=400,
            title="Integrand eˣ·√(1+eˣ) and Its Analytic Antiderivative"
        )
        .interactive()
    )

    int_chart
    return


app._unparsable_cell(
    r"""
    **Analytical Solution**

    \[
    \int e^{x}\sqrt{1+e^{x}}\,dx
    = \frac{2}{3}\bigl(1+e^{x}\bigr)^{3/2}+C.
    \]

    The chart above visualizes the original function \(f(x)=e^{x}\sqrt{1+e^{x}}\) (blue) and its antiderivative   
    \(F(x)=\frac{2}{3}(1+e^{x})^{3/2}\) (orange).
    """,
    name="_"
)


@app.cell(column=1)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from scipy.integrate import trapezoid, simpson

    from fuz import lint
    import numpy as np
    import polars as pl
    import altair as alt
    from fuz.dists import Beta
    from lintegrate import lqag, lqng, lcquad, logtrapz
    return (
        Beta,
        alt,
        lcquad,
        lint,
        logtrapz,
        lqag,
        lqng,
        np,
        pl,
        simpson,
        trapezoid,
    )


if __name__ == "__main__":
    app.run()
