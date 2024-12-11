import marimo

__generated_with = "0.9.32"
app = marimo.App(width="medium", app_title="ranking demo")


@app.cell
def __(mo):
    mo.md(
        r"""
        # better bayesian ranking with **fuz**

        <p align='center'>by miraia s chiou © 2024</p>

        ## introduction

        **fuz** can be used to improve on basic bayesian ranking by taking into account the shape of distributions rather than just the means.
        """
    )
    return


@app.cell
def __(np):
    seed = 42
    rng = np.random.default_rng(seed)
    return rng, seed


@app.cell
def __(mo):
    mo.md(
        r"""
        here is a description of the experiment to compare ranking systems.

        1. generate a list of scored items. per item:
            1. create reasonable sample scores (between 0 and 1 to make things easier)
            1. create reasonable sample counts
        1. from scored items, find a per-item distribution of possible true means
        1. create a simulation to find mean rankings
        """
    )
    return


@app.cell
def __(
    mo,
    w_init_n_scores,
    w_init_possible_means,
    w_n_items,
    w_xax_resolution,
):
    w_init_params = mo.vstack(
        [
            mo.md('## experiment parameters'),
            mo.hstack([w_n_items, w_init_possible_means], widths='equal', align='center'),
            mo.hstack([w_init_n_scores, w_xax_resolution], widths='equal', align='center'),
        ]
    )
    mo.callout(w_init_params, kind='info')
    return (w_init_params,)


@app.cell
def __(
    alt,
    fdb,
    mo,
    np,
    pl,
    w_init_n_scores,
    w_init_possible_means,
    w_n_items,
    w_xax_resolution,
):
    n_items = w_n_items.value
    _lb, _ub = w_init_possible_means.value
    init_possible_means = np.linspace(_lb, _ub, n_items)[:, None]
    xax = np.linspace(0, 1, w_xax_resolution.value)
    _al, _be = fdb.ab_from_mode_trials(0.06, 4)
    min_score_count, max_score_count = w_init_n_scores.value
    _scale = max_score_count - min_score_count
    score_dist = fdb.Beta(_al, _be, loc=min_score_count, scale=_scale)


    _x_count = np.linspace(min_score_count, max_score_count, 200)
    _chart = (
        alt.Chart(pl.DataFrame({'count': _x_count, 'density': score_dist.pdf(_x_count)}))
        .mark_line()
        .encode(x='count', y='density')
        .properties(title='score count distribution')
    )
    mo.ui.altair_chart(_chart)
    return (
        init_possible_means,
        max_score_count,
        min_score_count,
        n_items,
        score_dist,
        xax,
    )


@app.cell
def __(
    init_possible_means,
    max_score_count,
    n_items,
    np,
    rng,
    score_dist,
    stats,
):
    score_counts = score_dist.rvs((n_items, 1), random_state=rng).round().astype(np.int64)
    init_samp = stats.bernoulli.rvs(init_possible_means, size=(n_items, max_score_count))
    heads = np.take_along_axis(init_samp.cumsum(axis=1), score_counts - 1, axis=1)
    heads, score_counts
    return heads, init_samp, score_counts


@app.cell
def __():
    return


@app.cell
def __(n_items, score_dist):
    score_dist.rvs((n_items, 1)).dtype
    return


@app.cell
def __(mo):
    mo.md(r"""## code navigation""")
    return


@app.cell
def __(mo):
    mo.md(r"""### widgets""")
    return


@app.cell
def __(mo):
    mo.md(r"""#### initial parameters""")
    return


@app.cell
def __(mo):
    w_n_items = mo.ui.slider(
        steps=[5, 10, 50, 100, 500, 1000],
        debounce=True,
        label='no. of scored items',
        full_width=True,
        show_value=True,
        value=10,
    )
    w_init_possible_means = mo.ui.range_slider(
        0.01,
        0.99,
        0.01,
        value=(0.05, 0.95),
        debounce=True,
        show_value=True,
        label='possible mean scores',
        full_width=True,
    )
    w_init_n_scores = mo.ui.range_slider(
        2,
        200,
        1,
        value=(3, 36),
        debounce=True,
        label='possible no. of scores',
        full_width=True,
        show_value=True,
    )
    w_xax_resolution = mo.ui.slider(
        steps=[129, 257, 513, 1025, 2049, 4097, 8193],
        debounce=True,
        label='x axis resolution',
        full_width=True,
        show_value=True,
        value=1025,
    )
    return (
        w_init_n_scores,
        w_init_possible_means,
        w_n_items,
        w_xax_resolution,
    )


@app.cell
def __(mo):
    mo.md(r"""### imports""")
    return


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    import seaborn as sns
    import fuz.dists.beta as fdb
    return alt, fdb, mo, np, pd, pl, plt, sns, stats


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
