import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from scipy.integrate import trapezoid

    from fuz import lint
    import numpy as np
    import polars as pl
    import altair as alt
    from fuz.dists import Beta
    return Beta, alt, lint, np, pl, trapezoid


@app.cell(hide_code=True)
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
    doing numerical integration in logarithmic space allows one to be significantly more accurate when it comes to very small or very large numbers, which tend to push the limits of floating-point accuracy.
    """)
    return


@app.cell
def _(Beta):
    beta1 = Beta(9, 4)
    return (beta1,)


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
    def expf1(x):
        return np.exp(x) * np.sqrt(1 + np.exp(x))
    
    def expf1_int(x): 
        """Get the analytical indefinite integral of expf1"""
        return (2/3) * np.pow(1+np.exp(x), 1.5)

    def expf1_def_int(x1, x2):
        """Get the analytical definite integral of expf1 from x1 to x2."""
        return expf1_int(x2) - expf1_int(x1)

    return expf1, expf1_def_int


@app.cell
def _(expf1, expf1_def_int, np, trapezoid):
    expf1_lb, expf1_ub = 25, 30
    expf1_dx = 0.001
    expf1_x = np.arange(expf1_lb,expf1_ub,0.001)
    expf1_y = expf1(expf1_x)
    trapezoid(expf1_y, expf1_x), expf1_def_int(expf1_lb,expf1_ub), 
    return expf1_dx, expf1_x


@app.cell
def _(np):
    def lexpf1(x):
        """Log of expf1."""
        return x + 0.5*np.log(np.exp(x)+1) 
    return (lexpf1,)


@app.cell
def _(expf1_dx, expf1_x, lexpf1, lint, np):
    lexpf1_y = lexpf1(expf1_x)
    np.exp(lint.ltrap(lexpf1_y, expf1_dx))
    return


@app.cell
def _():
    import sympy as sp

    # Define symbol
    x = sp.symbols('x', real=True)

    # Original function expf1(x) = e^x * sqrt(1 + e^x)
    expf1_expr = sp.exp(x) * sp.sqrt(1 + sp.exp(x))

    # Logarithm of expf1
    log_expf1 = sp.log(expf1_expr)

    sp.simplify(log_expf1)
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


if __name__ == "__main__":
    app.run()
