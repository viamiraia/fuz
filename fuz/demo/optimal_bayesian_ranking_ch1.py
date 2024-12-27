import marimo

__generated_with = "0.10.7"
app = marimo.App(app_title="optimal bayesian ranking ch1")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # optimal bayesian ranking
        <h3 align='center'>chapter 1: bernoulli to beta</h3>
        <p align='center'>by miraia s. chiou © 2024</p>

        ## introduction

        **fuz** can be used to improve on basic bayesian ranking by taking into account the shape of distributions rather than just the means. the optimal method is to use mode-parameterized beta or dirichlet distributions, but using a prior obtained by multiplicative pooling also improves upon traditional [bayesian averaging/ranking](https://en.wikipedia.org/wiki/Bayesian_average)

        if you know already know how to derive the [rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession), skip to [claims](#claims). note that this chapter covers known principles. the next chapters cover research that could be considered novel (disclaimer: i am an amateur in this field).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## the bernoulli distribution

        to understand optimal bayesian ranking, start with the bernoulli distribution. this can be thought of as coin flips, upvotes and downvotes, yes/no ratings, or anything representable as binary. as an example the video game service _steam_ uses yes/no (thumbs up or thumbs down). in this case, a game with 6 thumbs up and 2 thumbs down has a bernoulli score distribution with $p=0.75$.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    from scipy.stats import bernoulli
    return (bernoulli,)


@app.cell(hide_code=True)
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
    _chart = (
        plot_bernoulli(intronoulli_p)
        .properties(width='container')
        .configure_mark(color='darkgreen')
    )
    mo.ui.altair_chart(_chart)
    return intronoulli, intronoulli_p


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""this distribution represents the sample mean. however, we have some additional information we want to represent - the number of ratings (count). we can do this with the binomial distribution.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## bernoulli to binomial""")
    return


@app.cell(hide_code=True)
def _():
    from scipy.stats import binom
    return (binom,)


@app.cell(hide_code=True)
def _(mo):
    wintro_no = mo.ui.slider(
        0, 10, value=2, show_value=True, label='no. of no/tails/downvotes', full_width=True
    )
    wintro_yes = mo.ui.slider(
        0, 10, value=6, show_value=True, label='no. of yes/heads/upvotes', full_width=True
    )
    mo.callout(mo.hstack([wintro_no, wintro_yes], widths='equal', align='center'), kind='info')
    return wintro_no, wintro_yes


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    alt,
    binom,
    intro2_n,
    intro2_p,
    plot_bernoulli,
    plot_binomial,
    wintro_no,
    wintro_yes,
):
    _noulli_chart = plot_bernoulli(intro2_p).properties(title='bernoulli pmf', width=100)


    intronomial = binom(intro2_n, intro2_p[1])
    _nomial_chart = plot_binomial(intronomial).properties(
        title=alt.TitleParams(
            text='binomial pmf',
            subtitle=f'n={intro2_n}, heads={wintro_yes.value}, tails={wintro_no.value}',
        ),
        width=300,
    )
    (_nomial_chart | _noulli_chart).interactive()
    return (intronomial,)


@app.cell(hide_code=True)
def _(mo, wintro_no, wintro_yes):
    mo.md(rf"""
    this is how the binomial distribution is typically first taught. the pmf shows the probability of $h$ successes, given $n$ trials. however, for bayesian ranking, this isn't what we need. 

    what we really want is to quantify the uncertainty around possible true scores. i.e., given {wintro_yes.value} heads and {wintro_no.value} tails, what's the probability that once we have $\infty$ ratings, the score will be $x$?

    good news - we can still start with the binomial distribution!

    we use the formula of the binomial pmf to hold $k$ and $n$ constant, sweeping $x$ to get probabilities and creating a chart.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## binomial to beta""")
    return


@app.cell(hide_code=True)
def _(mo):
    w_xlen = mo.ui.slider(5, 50, value=5, full_width=True, label='slide to sample!')
    mo.callout(w_xlen, kind='danger')
    return (w_xlen,)


@app.cell(hide_code=True)
def _(np, w_xlen):
    x_interactive = np.linspace(0, 1, w_xlen.value)
    return (x_interactive,)


@app.cell(hide_code=True)
def _(alt, binom, intro2_n, pd, wintro_no, wintro_yes, x_interactive):
    _df = pd.DataFrame(
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    from scipy.integrate import quad

    import fuz.dists as fd
    return fd, quad


@app.cell(hide_code=True)
def _(np):
    x_small = np.linspace(0, 1, 257)
    return (x_small,)


@app.cell(hide_code=True)
def _(Callable, Sequence, alt, np, pd):
    def plot_betas(x: np.ndarray, pdfs: Sequence[Callable], names: Sequence[str]) -> alt.Chart:
        dfs = []
        for pdf, name in zip(pdfs, names, strict=True):
            bdf = pd.DataFrame({'x': x, 'pdf': pdf(x)})
            bdf['name'] = name
            dfs.append(bdf)
        df = pd.concat(dfs)
        base = alt.Chart(df, title=name).encode(
            alt.X('x'), alt.Y('pdf'), alt.Color('name'), alt.StrokeDash('name')
        )
        return base.mark_line(opacity=0.5, strokeWidth=9)
    return (plot_betas,)


@app.cell(hide_code=True)
def _(
    alt,
    binom,
    fd,
    intro2_n,
    intro2_p,
    mo,
    pd,
    plot_betas,
    quad,
    wintro_yes,
    x_small,
):
    pre_pdf = lambda x: binom.pmf(wintro_yes.value, intro2_n, x)
    bin_pdf = lambda x: pre_pdf(x) / quad(pre_pdf, 0, 1)[0]
    _bin_df = pd.DataFrame({'x': x_small, 'pdf': bin_pdf(x_small)})
    _bin_df['name'] = 'binomial-derived'
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        from the chart, it's clear that the beta distribution parameterized by mode and number of trials matches the binomial-derived pdf.

        what are the implications? moving from bernoulli to binomial to beta is a well-known fundamental concept in statistics, so what's new here? let's set things up to understand.
        """
    )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""we have just derived the [rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession), aka the [bayes estimator](https://en.wikipedia.org/wiki/Binomial_distribution#Estimation_of_parameters) given a uniform prior. from here on i will refer to this estimator as $\mu_\Mu$ to reflect its meaning in the context of bayesian ranking.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## claims""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    1. in a bayesian ranking context, taking $\mu_\Mu$ is equivalent to finding the posterior, where the prior consists of the other items' possible true mean distributions $M_i$.
    1. this is extensible other rating systems like 3-star, 5-star, out-of-10, and even continuous (floating-point) systems, by using dirichlet distributions.
    1. given the same information, finding $\mu_\Mu$ is optimal and outperforms other bayesian ranking algorithms [^same-level]

    [^same-level]: caveat being algorithms on the same level; i'm not comparing to complex recommender systems, although it could act as a basis for a better recommender system since incorporating weights is simple.
    """),
        kind='warn',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        i will address these claims in the following chapters.

        although the rule of succession has been known since the 18th century, i believe this is a novel angle to look at the problem of bayesian ranking.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## code navigation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### widgets""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### initial parameters""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### plotting""")
    return


@app.cell(hide_code=True)
def _(Sequence, alt, np, pd):
    def plot_bernoulli(p: Sequence[float, float] | np.ndarray) -> alt.Chart:
        _df = pd.DataFrame({'x': ['negative', 'positive'], 'probability': p})
        _base = alt.Chart(_df, title='bernoulli probability mass function (pmf)').encode(
            alt.X('x', axis=alt.Axis(labelAngle=0)),
            alt.Y('probability', scale=alt.Scale(domain=[0, 1])),
        )
        _bar = _base.mark_bar()
        _text = _base.mark_text(dy=-10).encode(alt.Text('probability', format='.2f'))
        return _bar + _text
    return (plot_bernoulli,)


@app.cell(hide_code=True)
def _(alt, np, pd, rv_discrete_frozen):
    def plot_binomial(dist: rv_discrete_frozen, max_n: int = 20) -> alt.Chart:
        x = np.arange(max_n + 1)
        df = pd.DataFrame({'successes': x, 'probability': dist.pmf(x)})
        base = alt.Chart(df, title='binomial pmf').encode(
            alt.X('successes'), alt.Y('probability', scale=alt.Scale(domain=[0, 1]))
        )
        line = base.mark_line()
        point = base.mark_point()
        return line + point
    return (plot_binomial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### imports""")
    return


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

    import fuz.log as flog
    return alt, flog, mo, np, pd, rv_continuous_frozen, rv_discrete_frozen


if __name__ == "__main__":
    app.run()
