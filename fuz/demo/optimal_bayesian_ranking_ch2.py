import marimo

__generated_with = "0.10.6"
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

        we'll be looking at some ternary plots. if you haven't seen them before, this [tutorial](https://grapherhelp.goldensoftware.com/Graphs/Reading_Ternary_Diagrams.htm) may help. below i have created a plot of the three axes
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mpltern
    import numpy as np

    import fuz.log as flog
    return flog, mo, mpltern, np, plt


@app.cell
def _(mo, w_ternr, w_ternt):
    mo.callout(
        mo.vstack(
            [
                mo.md('### move the point on the plot'),
                mo.hstack([mo.md('$r$ axis'), w_ternr], widths=[1, 5], align='center'),
                mo.hstack([mo.md('$t$ axis'), w_ternt], widths=[1, 5], align='center'),
            ]
        ),
        kind='info',
    )
    return


@app.cell
def _(mo, w_ternr, w_ternt):
    ternr = w_ternr.value
    ternt = w_ternt.value
    ternl = round(1 - w_ternr.value - w_ternt.value, 2)
    mo.callout(mo.md(f"""$r,t,l = \\{{{ternr}, {ternt}, {ternl}\\}}$"""))
    return ternl, ternr, ternt


@app.cell
def _(np, plt, ternl, ternr, ternt):
    x_small = np.linspace(0, 1, 129)
    x_other = (1 - x_small) / 2
    dx_small = x_small[1] - x_small[0]
    p1 = np.vstack([x_small, x_other, x_other]).T
    p2 = np.vstack([x_other, x_small, x_other]).T
    p3 = np.vstack([x_other, x_other, x_small]).T
    _fig = plt.figure(figsize=(5, 4))
    _fig.subplots_adjust(top=0.8, bottom=0.15)
    _ax = plt.subplot(projection='ternary')
    _ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], color='steelblue', label='top axis')
    _ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], color='darkorange', label='left axis')
    _ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color='forestgreen', label='right axis')
    _ax.scatter(ternt, ternl, ternr, color='crimson')
    _ax.set_tlabel('$t$')
    _ax.set_llabel('$l$')
    _ax.set_rlabel('$r$')
    _ax.taxis.set_label_position('tick1')
    _ax.laxis.set_label_position('tick1')
    _ax.raxis.set_label_position('tick1')
    _ax.legend(fontsize=9, framealpha=0.3)
    _ax.grid(alpha=0.4)
    _fig
    return dx_small, p1, p2, p3, x_other, x_small


@app.cell
def _(mo):
    mo.md(
        r"""
        ## multinomial to dirichlet

        let's start by setting the number of ratings for an item in the 3-star rating system. in a sense, this is a 3-dimensional system. the three dimensions are 1-star, 2-star, and 3-star, or $\{0, 0.5, 1\}$.

        in a 5-star rating system, the five dimensions could be 1-5 or $\{0, 0.25, 0.5, 0.75, 1\}$
        """
    )
    return


@app.cell(hide_code=True)
def _():
    from scipy import stats

    import fuz.dists as fd
    import fuz.marimo as fmo
    import fuz.plot as fp
    return fd, fmo, fp, stats


@app.cell
def _(fmo, mo):
    w_3star_stack, (w_3star1, w_3star2, w_3star3) = fmo.make_star_widget(
        n_stars=3, star0=(1, 2, 3), max_ratings=12
    )
    mo.callout(w_3star_stack, kind='success')
    return w_3star1, w_3star2, w_3star3, w_3star_stack


@app.cell
def _(np, w_3star1, w_3star2, w_3star3):
    trials_3star = np.array([w_3star1.value, w_3star2.value, w_3star3.value])
    alpha_3star = trials_3star + 1
    return alpha_3star, trials_3star


@app.cell
def _(flog, stats, trials_3star):
    n_3star = trials_3star.sum()
    p_3star = flog.norm(trials_3star)
    multi_3star = stats.multinomial(n=n_3star, p=p_3star)
    return multi_3star, n_3star, p_3star


@app.cell
def _(mo):
    mo.md(r"""in this system, the analogous distribution to the binomial is the multinomial:""")
    return


@app.cell
def _(fp, multi_3star):
    _fig, _ax = fp.plot_multinomial(multi_3star, '3-star multinomial pmf')
    _fig.set_figwidth(5)
    _fig.set_figheight(4)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""the analogous distribution to the beta is the dirichlet. it is characterized by $\alpha$, a vector of the number of ratings + 1 for each dimension.""")
    return


@app.cell
def _(alpha_3star, fd, fp):
    d_3star = fd.Scored(alpha_3star, [1, 2, 3])
    _fig, _ax = fp.plot_scored_pdf(d_3star, '3-star dirichlet pdf')
    _fig.set_figwidth(5)
    _fig.set_figheight(4)
    _fig
    return (d_3star,)


@app.cell
def _(mo):
    mo.md(
        r"""
        just like in chapter 1, you can get the dirichlet from the multinomial. it's harder to visualize but sweeping $p$ along top, left, and right cross-sections of the ternary plot, you can see that the distributions match. 

        the multinomial-derived is the opaque line, while the corresponding dirichlet plot is the translucent thick line.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import altair as alt
    import pandas as pd
    from scipy.integrate import romb
    return alt, pd, romb


@app.cell
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

        note that this works for the binary case. working backwards:

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


@app.cell
def _(mo):
    mo.md(
        r"""
        ## reaching infinity

        what if you want to allow for arbitrary ratings? ex. instead of a 4 star rating, allow for 4.23. we can extend the dirichlet dimensions to infinity and see what happens.
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


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
        no matter what, the bayes estimator / rule of succession approaches $0.5$ as the number of dimensions approaches $\infty$.

        this is not that useful then. can we think of something else that may help? let's start by subtracting 0.5, so it's not centered on 0.5, and multiply by the number of dimensions to normalize it.
        """
    )
    return


@app.cell
def _(alt, infdf1, mo):
    _chart = (
        alt.Chart(infdf1).mark_line().encode(alt.X('dimensions'), alt.Y('y').scale(zero=False))
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        this is a lot better! now this approaches a value, and can potentially be used for ranking. after some experimentation, i figured out the asymptote.

        $$
        \begin{equation}
        (\textrm{mode}-0.5) (\alpha_i-1)
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(mo, w_infcnt, w_infval):
    mo.md(rf"""so for our current floating-point score of ${w_infval.value}$ and score count of ${w_infcnt.value}$, the asymptote is 

    $$
    {(w_infval.value - 0.5)*(w_infcnt.value):0.5g}
    $$""")
    return


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""how is this useful? note that since $\mu$ is the bayes estimator, all we did was move and scale it, so it's essentially a linear transformation. thus our new infinite estimator can still be used for ranking!""")
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
    w_ternr = mo.ui.slider(0, 1, 0.01, value=0.5, full_width=True, show_value=True)
    mo.accordion({'ternary r widget': w_ternr})
    return (w_ternr,)


@app.cell
def _(mo):
    get_t, set_t = mo.state(0)
    return get_t, set_t


@app.cell
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
    # common imports
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
