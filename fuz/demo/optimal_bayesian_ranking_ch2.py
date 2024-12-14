import marimo

__generated_with = "0.10.2"
app = marimo.App(width="medium", app_title="optimal bayesian ranking ch1")


@app.cell
def _(mo):
    mo.md(
        r"""
        # optimal bayesian ranking
        <h3 align='center'>chapter 1: bernoulli to beta</h3>
        <p align='center'>by miraia s. chiou © 2024</p>

        ## introduction

        **fuz** can be used to improve on basic bayesian ranking by taking into account the shape of distributions rather than just the means. the optimal method is to use mode-parameterized beta or dirichlet distributions, but using a prior obtained by multiplicative pooling (aka upco) also improves upon traditional [bayesian averaging/ranking](https://en.wikipedia.org/wiki/Bayesian_average)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## the bernoulli distribution

        to understand how my optimal bayesian ranking algorithm works, start with the bernoulli distribution. this can be thought of as coin flips, upvotes and downvotes, yes/no ratings, or anything representable as binary. as an example the video game service _steam_ uses yes/no (thumbs up or thumbs down). in this case, a game with 6 thumbs up and 2 thumbs down has a bernoulli score distribution with $p=0.75$.
        """
    )
    return


@app.cell
def _():
    from scipy.stats import bernoulli
    return (bernoulli,)


@app.cell
def _(mo):
    wintro_score = mo.ui.slider(
        0, 1, 0.01, value=0.75, show_value=True, label='choose a score', full_width=True
    )
    mo.callout(wintro_score, kind='success')
    return (wintro_score,)


@app.cell
def _(bernoulli, mo, plot_bernoulli, wintro_score):
    intronoulli = bernoulli(wintro_score.value)
    intronoulli_p = [intronoulli.pmf(0), intronoulli.pmf(1)]
    _chart = plot_bernoulli(intronoulli_p).properties(width=400).configure_mark(color='darkgreen')
    mo.ui.altair_chart(_chart)
    return intronoulli, intronoulli_p


@app.cell
def _(mo):
    mo.md(r"""this distribution represents the sample mean. however, we have some additional information we want to represent - the number of ratings (count). we can do this with the binomial distribution.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## bernoulli to binomial""")
    return


@app.cell
def _():
    from scipy.stats import binom
    return (binom,)


@app.cell
def _(mo):
    wintro_no = mo.ui.slider(
        0, 10, value=2, show_value=True, label='\# of no/tails/downvotes', full_width=True
    )
    wintro_yes = mo.ui.slider(
        0, 10, value=6, show_value=True, label='\# of yes/heads/upvotes', full_width=True
    )
    mo.callout(mo.hstack([wintro_no, wintro_yes], widths='equal', align='center'), kind='info')
    return wintro_no, wintro_yes


@app.cell
def _(flog, mo, np, wintro_no, wintro_yes):
    intro2_weights = np.array([wintro_no.value, wintro_yes.value])
    intro2_p = flog.norm(intro2_weights)
    intro2_n = intro2_weights.sum()
    mo.md(f"""
    here i introduce `fuz.log.lnorm` which can stably normalize weights in logarithmic space (log -> log), handling `nan`s and complex numbers. a little excessive for our example here, but works well with very small probabilities. if you want to move out of log space, a convenience function, `norm`, does log conversion and exponentiation for you.

    ```python
    import fuz.log as flog
    import numpy as np

    flog.norm({intro2_weights}) # {intro2_p}
    # this is equivalent to:
    np.exp(flog.lnorm(np.log({intro2_weights}))) # {intro2_p}
    ```

    """)
    return intro2_n, intro2_p, intro2_weights


@app.cell
def _(
    alt,
    binom,
    intro2_n,
    intro2_p,
    mo,
    plot_bernoulli,
    plot_binomial,
    wintro_no,
    wintro_yes,
):
    intronomial = binom(intro2_n, intro2_p[1])
    _noulli_chart = plot_bernoulli(intro2_p)
    _nomial_chart = plot_binomial(intronomial).properties(
        title=alt.TitleParams(
            text='binomial pmf',
            subtitle=f'n={intro2_n}, heads={wintro_yes.value}, tails={wintro_no.value}',
        )
    )
    mo.hstack([mo.ui.altair_chart(_noulli_chart), mo.ui.altair_chart(_nomial_chart)])
    return (intronomial,)


@app.cell
def _(mo, wintro_no, wintro_yes):
    mo.md(f"""
    this is how the binomial distribution is typically first taught. the pmf shows the probability of $h$ successes, given $n$ trials. however, for bayesian ranking, this isn't what we need. 

    what we really want is to quantify the uncertainty around possible true scores. i.e., given {wintro_yes.value} heads and {wintro_no.value} tails, what's the probability that once we have $\infty$ ratings, the score will be $x$?

    good news - we can still start with the binomial distribution!

    we use the formula of the binomial pmf to hold $k$ and $n$ constant, sweeping $x$ to get probabilities and creating a chart.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## binomial to beta""")
    return


@app.cell
def _(mo):
    w_xlen = mo.ui.slider(5, 50, value=5, full_width=True, label='slide to sample!')
    mo.callout(w_xlen, kind='danger')
    return (w_xlen,)


@app.cell
def _(np, w_xlen):
    x_interactive = np.linspace(0, 1, w_xlen.value)
    return (x_interactive,)


@app.cell
def _(alt, binom, intro2_n, pl, wintro_no, wintro_yes, x_interactive):
    _df = pl.DataFrame(
        {
            'true mean': x_interactive,
            'probability': binom.pmf(wintro_yes.value, intro2_n, x_interactive),
        }
    )
    _base = alt.Chart(
        _df,
        title=alt.TitleParams(
            text='probability of potential true means',
            subtitle=f'given {wintro_yes.value} heads and {wintro_no.value} tails',
        ),
    ).encode(alt.X('true mean'), alt.Y('probability'))
    _line = _base.mark_line(color='maroon')
    _point = _base.mark_point(color='maroon')
    _chart = _line + _point
    _chart.properties(width=600)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        what do you notice here? 

        1. this is not a probability distribution function (pdf) yet. to make it a pdf, we'll divide the probability by the integral to get the density.
        1. the mode (peak) is $p$.
        1. the mean of the possible true means is different from the mode.
        1. after turning this into a pdf, this is a beta distribution.

        let's turn this into a pdf and see which beta distributions might match. fortunately **fuz** allows you to make beta distributions in a variety of ways.
        """
    )
    return


@app.cell
def _():
    from scipy.integrate import quad
    import fuz.dists as fd
    return fd, quad


@app.cell
def _(np):
    x_small = np.linspace(0, 1, 257)
    return (x_small,)


@app.cell
def _(Callable, Sequence, alt, np, pl):
    def plot_betas(x: np.ndarray, pdfs: Sequence[Callable], names: Sequence[str]) -> alt.Chart:
        dfs = []
        for pdf, name in zip(pdfs, names, strict=True):
            bdf = pl.DataFrame({'x': x, 'pdf': pdf(x)}).with_columns(name=pl.lit(name))
            dfs.append(bdf)
        df = pl.concat(dfs)
        base = alt.Chart(df, title=name).encode(
            alt.X('x'), alt.Y('pdf'), alt.Color('name'), alt.StrokeDash('name')
        )
        return base.mark_line(opacity=0.5, strokeWidth=9)
    return (plot_betas,)


@app.cell
def _(
    alt,
    binom,
    fd,
    intro2_n,
    intro2_p,
    mo,
    pl,
    plot_betas,
    quad,
    wintro_yes,
    x_small,
):
    pre_pdf = lambda x: binom.pmf(wintro_yes.value, intro2_n, x)
    bin_pdf = lambda x: pre_pdf(x) / quad(pre_pdf, 0, 1)[0]
    _bin_df = pl.DataFrame({'x': x_small, 'pdf': bin_pdf(x_small)}).with_columns(
        name=pl.lit('binomial-derived')
    )
    _bin_fig = (
        alt.Chart(_bin_df, title='binomial-derived vs potential betas')
        .mark_line()
        .encode(alt.X('x'), alt.Y('pdf'), alt.Color('name'), alt.StrokeDash('name'))
    )

    _mo = intro2_p[1]
    b_mo_t = fd.beta_from_mode_trials(_mo, intro2_n)
    b_mo_k1 = fd.beta_from_mode_k(_mo, intro2_n)
    b_mu_k1 = fd.beta_from_mu_k(_mo, intro2_n)
    _beta_fig = plot_betas(
        x_small, (b_mo_t.pdf, b_mo_k1.pdf, b_mu_k1.pdf), ('β mode trials', 'β mode k', 'β μ k')
    )

    mo.ui.altair_chart(_bin_fig + _beta_fig)
    return b_mo_k1, b_mo_t, b_mu_k1, bin_pdf, pre_pdf


@app.cell
def _(mo):
    mo.md(
        r"""
        from the chart, it's clear that the beta distribution parameterized by mode and number of trials matches the binomial-derived pdf.

        what are the implications? moving from bernoulli to binomial to beta is a well-known fundamental concept in statistics, so what's new here? let's set things up to understand.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## setup

        first, for the purposes of this chapter, let's define:

        $$
        \begin{aligned}
        \mu_s &= \text{sample mean} \\
        \mu_\top &= \text{the true mean} \\
        \mu_\diamond &= \text{a possible true mean} \\
        \Mu &= \text{the distribution of possible true means} \\
        \mathrm{E}[M] = \mu_\Mu &= \text{the mean of possible true means} \\
        \hat{\Mu} &= \text{the mode of the possible true mean distribution} \\
        \hat{\Beta} &= \text{the mode of a beta distribution} \\
        \end{aligned}
        $$

        in addition, we use standard notation for the [beta distribution from wikipedia](https://en.wikipedia.org/wiki/Beta_distribution).

        here are some insights:

        1. the parameterization of the beta by mode and no. of trials $t$, where $t = k-2$, matches best.
        1. the mean of the beta $\mu_B$ corresponds to the mean of possible true means $\mu_\Mu$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## claims""")
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("""

    1. in a ranking context, taking $\mu_\Mu$ is equivalent to finding the posterior, where the prior consists of the other items' possible true mean distributions $M_i$.
    1. this is extensible other rating systems like 3-star, 5-star, out-of-10, and even continuous (floating-point) systems, by using dirichlet distributions.
    1. given the same information, finding $\mu_\Mu$ is optimal and outperforms other bayesian ranking algorithms [^same-level]

    [^same-level]: caveat being algorithms on the same level; i'm not comparing to complex recommender systems, although it could act as a basis for a better recommender system since incorporating weights is simple.
    """),
        kind='warn',
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        i will address these claims in the following chapters.

        i believe this is a novel angle to look at the problem of bayesian ranking, although i wouldn't be surprised if someone has done something similar before.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## the mean of possible true means

        first, what's the best way to calculate $\mu_\Mu$? just a bit of algebra:

        $$
        \begin{aligned}
        t &= k-2 \\
        k &= \alpha + \beta \\
        \hat{\Beta} &\coloneqq \mu_s\\
        \hat{\Beta} &= \frac{\alpha-1}{\alpha+\beta-2} = \frac{\alpha-1}{t} \\
        \Mu &= \Beta(\alpha,\beta) \\
        \mu_\Mu &= \frac{\alpha}{k} \\
        \alpha &= k\mu_\Mu = t\hat{\Beta} + 1 \\
        \mu_\Mu &= \frac{t \mu_s + 1}{k} \\
        \end{aligned}
        $$

        thus, to find $\mu_\Mu$ given number of ratings $t$ and mean $\mu_s$:
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
    $$
    \mu_\Mu = \frac{t \mu_s + 1}{t+2}
    $$
    """),
        kind='success',
    )
    return


@app.cell
def _():
    return


@app.cell
def _(np):
    seed = 42
    rng = np.random.default_rng(seed)
    return rng, seed


@app.cell
def _(mo):
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
def _(
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
def _(
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
def _(
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
def _():
    return


@app.cell
def _(n_items, score_dist):
    score_dist.rvs((n_items, 1)).dtype
    return


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
def _(mo):
    mo.md(r"""### plotting""")
    return


@app.cell
def _(Sequence, alt, np, pl):
    def plot_bernoulli(p: Sequence[float, float] | np.ndarray) -> alt.Chart:
        _df = pl.DataFrame({'x': ['negative', 'positive'], 'probability': p})
        _base = alt.Chart(_df, title='bernoulli probability mass function (pmf)').encode(
            alt.X('x', axis=alt.Axis(labelAngle=0)),
            alt.Y('probability', scale=alt.Scale(domain=[0, 1])),
        )
        _bar = _base.mark_bar()
        _text = _base.mark_text(dy=-10).encode(alt.Text('probability', format='.2f'))
        return _bar + _text
    return (plot_bernoulli,)


@app.cell
def _(alt, np, pl, rv_discrete_frozen):
    def plot_binomial(dist: rv_discrete_frozen, max_n: int = 20) -> alt.Chart:
        x = np.arange(max_n + 1)
        df = pl.DataFrame({'successes': x, 'probability': dist.pmf(x)})
        base = alt.Chart(df, title='binomial pmf').encode(alt.X('successes'), alt.Y('probability'))
        line = base.mark_line()
        point = base.mark_point()
        return line + point
    return (plot_binomial,)


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
    import fuz.dists.beta as fdb
    import fuz.log as flog
    from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
    return (
        alt,
        fdb,
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
