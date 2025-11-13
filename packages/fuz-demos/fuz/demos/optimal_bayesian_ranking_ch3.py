import marimo

__generated_with = '0.10.7'
app = marimo.App(app_title='optimal bayesian ranking ch3')


@app.cell(hide_code=True)
def _():
    import io

    import altair as alt
    import fuz.core.marimo as fmo
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.integrate import romb

    import fuz.lint as flog

    return alt, flog, fmo, io, mo, np, pd, plt, romb


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # optimal bayesian ranking
        <h3 align='center'>chapter 3: ranking comparisons</h3>
        <p align='center'>by miraia s. chiou © 2024</p>

        ## introduction

        in the previous chapter, i demonstrated how the rule of succession for a 3-star based scoring system. i showed my infinite version of the rule of succession, allowing for optimal ranking in a scoring sytem allowing for rational number ratings.

        in this chapter, i will use a novel method to compare various ranking methods. comparing ranking with uncertainty has some subtleties:

        - you should compare how well your ranking captures potential true means, not the true means themselves.
            - a common mistake is creating a single true score per item and seeing how well a ranking predicts the true non-bayesian ranking.
            - basically you need to compare both the estimate and the uncertainty.
        - how do you compare how well an uncertainty estimate captures the true uncertainty?
        """
    )
    return


@app.cell
def _(mo):
    w_google_data = mo.ui.file(
        filetypes=['.parquet', '.csv', '.csv.gz'],
        label='upload google local data here',
        kind='area',
    )
    w_google_data
    return (w_google_data,)


@app.cell
def _(io, pd, w_google_data):
    _file = io.BytesIO(w_google_data.contents())
    df = pd.read_parquet(_file)
    df
    return (df,)


@app.cell
def _(fmo):
    w_starcheck_layout, [w_star1, w_star2, w_star3, w_star4, w_star5] = fmo.make_star_widget()
    w_starcheck_layout
    return w_star1, w_star2, w_star3, w_star4, w_star5, w_starcheck_layout


@app.cell
def _(alt, df, w_star1, w_star2, w_star3, w_star4, w_star5):
    star_filt = (
        (df['1'] == w_star1.value)
        & (df['2'] == w_star2.value)
        & (df['3'] == w_star3.value)
        & (df['4'] == w_star4.value)
        & (df['5'] == w_star5.value)
    )

    alt.Chart(df[star_filt]).transform_density(
        'score',
        as_=['score', 'density'],
    ).mark_area().encode(
        alt.X('score', scale=alt.Scale(domain=(1, 5))),
        alt.Y('density:Q'),
    )
    return (star_filt,)


@app.cell
def _(df):
    len(df[df['1'] > 5]), len(df)
    return


@app.cell(hide_code=True)
def _(mo):
    w_seed = mo.ui.slider(
        0, 100, 1, value=42, full_width=True, show_value=True, label='random seed'
    )
    mo.callout(w_seed, kind='success')
    return (w_seed,)


@app.cell(hide_code=True)
def _(np, pd, w_seed):
    rng = np.random.default_rng(w_seed.value)
    bi_counts = rng.integers(2, 25, 10)
    bi_scores = rng.random(10)
    bi_df = (
        pd.DataFrame({'score': bi_scores, 'count': bi_counts})
        .sort_values('score', ascending=False)
        .reset_index(drop=True)
    )
    bi_df
    return bi_counts, bi_df, bi_scores, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## multinomial to dirichlet

        let's start by setting the number of ratings for an item in the 3-star rating system. in a sense, this is a 3-dimensional system. the three dimensions are 1-star, 2-star, and 3-star, or $\{0, 0.5, 1\}$.

        just as an example, in a 5-star rating system, the five dimensions could be 1-5 or $\{0, 0.25, 0.5, 0.75, 1\}$

        play with the rating counts below to see how the corresponding distributions change!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""in this system, the analogous distribution to the binomial is the multinomial. it can be represented by a ternary plot:"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## citations

        ### for the google reviews data

        **UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining**
        Jiacheng Li, Jingbo Shang, Julian McAuley
        Annual Meeting of the Association for Computational Linguistics (ACL), 2022
        pdf

        **Personalized Showcases: Generating Multi-Modal Explanations for Recommendations**
        An Yan, Zhankui He, Jiacheng Li, Tianyang Zhang, Julian Mcauley
        arXiv:2207.00422, 2022
        """
    )
    return


@app.cell
def _(alpha_3star, fd, fp):
    d_3star = fd.Scored(alpha_3star, [1, 2, 3])
    _fig, _ax = fp.plot_scored_pdf(d_3star, title='3-star dirichlet pdf')
    _fig.set_figwidth(5)
    _fig.set_figheight(4)
    _fig
    return (d_3star,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        just like in chapter 1, you can get the dirichlet from the multinomial. it's harder to visualize but sweeping $p$ along top, left, and right cross-sections of the ternary plot, you can see that the distributions match. again, we use a mode-based parameterization to get the matching distribution.

        the multinomial-derived pdf is the opaque line, while the corresponding dirichlet plot is the translucent thick line.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(
    alt,
    d_3star,
    dx_small,
    mo,
    n_3star,
    p1,
    p2,
    p3,
    pd,
    romb,
    stats,
    trials_3star,
    x_small,
):
    _ps = [p1, p2, p3]
    _dfs = []
    for _i, _p in enumerate(_ps):
        _ym = stats.multinomial.pmf(trials_3star, n=n_3star, p=_p)
        _ym = _ym / romb(_ym, dx_small)
        _yd = d_3star.pdf(_p.T)
        _yd = _yd / romb(_yd, dx_small)
        _df = pd.DataFrame({'x': x_small, 'multinomial': _ym, 'dirichlet': _yd})
        _df['axis'] = _i + 1
        _dfs.append(_df)
    _df = pd.concat(_dfs)
    _base = alt.Chart(_df).encode(
        alt.X('x'), alt.Color('axis:N', scale=alt.Scale(scheme='category10'))
    )
    _multi_chart = _base.mark_line().encode(y='multinomial')
    _diri_chart = _base.mark_line(opacity=0.2, strokeWidth=12).encode(y='dirichlet')
    _chart = alt.layer(_multi_chart, _diri_chart)
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        in our specific 3-star case, the rule of succession is:

        $$
        \begin{equation}
        \mu_\Mu = \frac{t_1 + 1}{t+3} + \frac{2(t_2 + 1)}{t+3} + \frac{3(t_3 + 1)}{t+3}
        \end{equation}
        $$

        as we expand to more dimensions / more stars, the general rule is:

        $$
        \begin{equation}
        \mu_\Mu = \sum_{d=1}^m\frac{x_d(t_d + 1)}{t + m}
        \end{equation}
        $$

        where

        $$
        \begin{aligned}
        \mu_\Mu &= \text{the mean of possible true means}\\
        N &= \text{number of dimensions} \\
        x_d &= \text{value of dimension } d\\
        t &= \text{total number of trials (ratings)} \\
        t_d &= \text{number of trials for dimension } d\\
        \end{aligned}
        $$

        note that this works for the binary case (ch1). working backwards:

        $$
        \begin{aligned}
        \mu_s &= \frac{0\cdot t_1 + 1 \cdot t_2}{t} \\
        &= \frac{t_2}{t} \\
        \mu_\Mu &= \frac {t\mu_s + 1}{t+2} \\
        &= \frac{t(t_2 / t)+1}{t+2}\\
        &= \frac{t_2 + 1}{t+2}\\
        &= \frac{0(t_1 + 1)}{t+2} + \frac{1(t_2 + 1)}{t+2}\\
        &= \sum_{d=1}^N\frac{x_d(t_d + 1)}{t + N} \\
        \end{aligned}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## reaching infinity

        what if you want to allow for arbitrary ratings? ex. instead of a 4 star rating, allow for 4.23, 3, 2.718281828... etc. we can gradually extend the dirichlet dimensions toward infinity and see what happens.

        to do this we create dimensions along the interval $[0,1]$ and increase the alpha for that dimension by one (using mode parameterization, meaning each alpha starts with 1). for visualization purposes we increase the dimensions by 100 each time and only allow user input wih a resolution of 0.01. then we try to figure out the mean, which would be the rule of succession...

        however...
        """
    )
    return


@app.cell
def _(fd, np):
    def get_like_infdir(vals, cnts, n_dim=1001):
        scores = np.linspace(0, 1, n_dim)
        alpha = np.ones(n_dim)
        for val, cnt in zip(vals, cnts):
            ind = round((n_dim - 1) * val)
            alpha[ind] += cnt
        return fd.Scored(alpha, scores)

    return (get_like_infdir,)


@app.cell(hide_code=True)
def _(mo):
    w_infval = mo.ui.slider(0.01, 0.99, 0.01, 0.01, full_width=True, show_value=True)
    w_infcnt = mo.ui.slider(1, 100, 1, 1, full_width=True, show_value=True)
    mo.callout(
        mo.vstack(
            [
                mo.md('### add a single value'),
                mo.hstack([mo.md('score'), w_infval], widths=[1, 5], align='center'),
                mo.hstack([mo.md('count'), w_infcnt], widths=[1, 5], align='center'),
            ]
        ),
        kind='warn',
    )
    return w_infcnt, w_infval


@app.cell(hide_code=True)
def _(alt, get_like_infdir, mo, np, pd, w_infcnt, w_infval):
    infx = np.arange(100, 10001, 100)
    _mus, _ys = [], []
    for _ndim in infx:
        d = get_like_infdir([w_infval.value], [w_infcnt.value], n_dim=_ndim)
        _mus.append(d.mu)
        _ys.append((d.mu - 0.5) * _ndim)

    infdf1 = pd.DataFrame({'dimensions': infx, 'mu': _mus, 'y': _ys})
    _chart = (
        alt.Chart(infdf1).mark_line().encode(alt.X('dimensions'), alt.Y('mu').scale(zero=False))
    )
    mo.ui.altair_chart(_chart)
    return d, infdf1, infx


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        no matter what, the bayes estimator / rule of succession approaches $0.5$ as the number of dimensions approaches $\infty$.

        this is not that useful then... what can we do?

        let's start by subtracting 0.5, so the estimator is not centered on 0.5, and multiply by the number of dimensions to normalize it.
        """
    )
    return


@app.cell(hide_code=True)
def _(alt, infdf1, mo):
    _chart = (
        alt.Chart(infdf1).mark_line().encode(alt.X('dimensions'), alt.Y('y').scale(zero=False))
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        this is a lot better! now this approaches a value, and can potentially be used for ranking. after some experimentation, i figured out the asymptote.

        $$
        \begin{align}
        \sum_{i=1}^N(\textrm{mode}_i-0.5)(\alpha_i-1)
        \end{align}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, w_infcnt, w_infval):
    mo.md(rf"""so for our current floating-point score of ${w_infval.value}$ and score count of ${w_infcnt.value}$, the asymptote is

    $$
    {(w_infval.value - 0.5) * (w_infcnt.value):0.5g}
    $$

    note that 0.5 in the equation is the midpoint of the scale 0-1.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        here's a derivation. note that this derivation may be wrong due to my lack of math skills, but i have empirically confirmed the result to be correct.

        $$
        \begin{aligned}
        \alpha_i &\ge 1 \\
        \alpha_0 &= \sum_{i=1}^N \alpha_i \\
        \mu(x_i) &= \sum_{i=1}^N \frac{\alpha_i}{\alpha_0}\cdot x_i \\
        &= \sum_{i=1}^N \frac{\alpha_i}{\sum_{i=1}^N \alpha_i}\cdot x_i\\
        m &= \text{scale midpoint} \\
        f(x_i) &= N\cdot(\mu(x_i) - m) \\
        \lim_{N \to \infty} f(x_i) &= \lim_{N \to \infty} N\cdot(\mu(x_i) - m)\\
        &= \lim_{N \to \infty}N \left( \sum_{i=1}^N \frac{\alpha_i}{\sum_{i=1}^N \alpha_i}\cdot x_i \right)-Nm \\
        &= \lim_{N \to \infty} N \left( \sum_{i=1}^N \frac{\alpha_i}{N \cdot \alpha_{avg}} \cdot x_i \right) - Nm \\
        &= \lim_{N \to \infty} N \left( \sum_{i=1}^N \frac{\alpha_i}{N \cdot \alpha_{\infty}} \cdot x_i \right) - Nm  \\
        &= \lim_{N \to \infty} \left( \sum_{i=1}^N \frac{\alpha_i}{\alpha_{\infty}} \cdot x_i \right) - Nm \\
        &= \lim_{N \to \infty} \left( \sum_{i=1}^N \frac{\alpha_i}{\alpha_{\infty}} \cdot x_i \right) - N \cdot m + N \cdot m - N \cdot m  \\
        &= \lim_{N \to \infty} \left( \sum_{i=1}^N \frac{\alpha_i}{\alpha_{\infty}} \cdot x_i - \frac{\alpha_i}{\alpha_{\infty}} \cdot m \right)  \\
        &= \lim_{N \to \infty} \sum_{i=1}^N \frac{\alpha_i}{\alpha_{\infty}} \cdot (x_i - m)\\
        &= \lim_{N \to \infty} \sum_{i=1}^N (\alpha_i - 1) \cdot (x_i - m)\\
        &= \sum_{i=1}^N(x_i-m)(\alpha_i-1)\\
        \end{aligned}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    #### miraia's infinite rule of succession / infinite-dimensional factor

    $$
    \begin{equation}
    \lim_{N \to \infty} f(x_i) = \sum_{i=1}^N(x_i-m)(\alpha_i-1)\\
    \end{equation}
    $$
    """),
        kind='success',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""how is this useful? we can use it for ranking! now that we're past the introductory concepts, in the next chapter we'll dive into optimal bayesian ranking."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## additional code""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### widgets""")
    return


@app.cell(hide_code=True)
def _(mo):
    w_ternr = mo.ui.slider(0, 1, 0.01, value=0.5, full_width=True, show_value=True)
    mo.accordion({'ternary r widget': w_ternr})
    return (w_ternr,)


@app.cell(hide_code=True)
def _(mo):
    get_t, set_t = mo.state(0)
    return get_t, set_t


@app.cell(hide_code=True)
def _(get_t, mo, set_t, w_ternr):
    _t = get_t()
    _max = 1 - w_ternr.value
    if _t > _max:
        _t = _max
    w_ternt = mo.ui.slider(
        0,
        1 - w_ternr.value,
        0.01,
        value=_t,
        full_width=True,
        show_value=True,
        on_change=lambda v: set_t(v),
    )
    mo.accordion({'ternary t widget': w_ternt})
    return (w_ternt,)


if __name__ == '__main__':
    app.run()
