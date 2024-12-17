import marimo

__generated_with = "0.10.2"
app = marimo.App(width="medium", app_title="optimal bayesian ranking ch2")


@app.cell
def _(mo):
    mo.md(
        r"""
        # optimal bayesian ranking
        <h3 align='center'>chapter 2: to infinity</h3>
        <p align='center'>by miraia s. chiou © 2024</p>

        ## introduction

        in the previous chapter, i demonstrated how to derive the [rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession), aka the [bayes estimator](https://en.wikipedia.org/wiki/Binomial_distribution#Estimation_of_parameters) with a uniform prior. in a ranking context, this represents the mean of the distribution of potential true means $\mu_\Mu = \dfrac{t \mu_s + 1}{t+2}$.

        in this chapter, i will show original research extending the estimator to the inifinite dirichlet case.
        """
    )
    return


@app.cell
def _():
    import fuz.rank as fr
    return (fr,)


@app.cell
def _(mo):
    mo.md(r"""## code navigation""")
    return


@app.cell
def _(mo):
    mo.md(r"""### widgets""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### initial parameters""")
    return


@app.cell
def _(mo):
    mo.md(r"""### plotting""")
    return


@app.cell
def _(mo):
    mo.md(r"""### imports""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    import seaborn as sns
    import fuz.dists as fd
    import fuz.log as flog
    from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
    return (
        alt,
        fd,
        flog,
        mo,
        np,
        pd,
        pl,
        plt,
        rv_continuous_frozen,
        rv_discrete_frozen,
        sns,
        stats,
    )


if __name__ == "__main__":
    app.run()
