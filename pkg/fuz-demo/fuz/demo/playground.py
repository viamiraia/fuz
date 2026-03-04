import marimo

__generated_with = "0.19.7"
app = marimo.App(width="columns", app_title="fuz playground")


@app.cell(column=0)
def _(mo):
    mo.md(r"""
    # `fuz` playground
    """)
    return


@app.cell(column=1)
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## distributions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### beta distribution
    """)
    return


@app.cell
def _(mo, w_beta_a, w_beta_b):
    mo.hstack([w_beta_a, w_beta_b])
    return


@app.cell
def _(mo, w_beta_mode, w_beta_trials):
    mo.hstack([w_beta_mode, w_beta_trials])
    return


@app.cell
def _(mo, w_beta_k, w_beta_mu, w_beta_var):
    mo.vstack([mo.hstack([w_beta_mu, w_beta_k]), w_beta_var])
    return


@app.cell
def _(beta, beta_xax, mo, plot_beta, w_beta_chart_height, w_beta_chart_width):
    chart_beta_pdf, chart_beta_cdf = plot_beta(
        beta_xax,
        beta,
        kind='sep',
        width=w_beta_chart_width.value,
        height=w_beta_chart_height.value,
    )
    mo.vstack([chart_beta_pdf, chart_beta_cdf], align='center')
    return


@app.cell
def _(mo, w_beta_chart_height, w_beta_chart_width):
    mo.hstack([w_beta_chart_width, w_beta_chart_height])
    return


@app.cell
def _(beta, mo, pl):
    mo.ui.table(
        pl.DataFrame(beta.stats).transpose(
            include_header=True, header_name='stat', column_names=['value']
        ),
        page_size=12,
        show_column_summaries=False,
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### dirichlet distribution
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### other distributions
    """)
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    import marimo as mo
    import altair as alt
    import fuz.dists as fd
    import fuz.types as ft
    import numpy as np
    import polars as pl

    from collections.abc import Callable
    from typing import Literal
    return Callable, Literal, alt, fd, ft, mo, np, pl


@app.cell
def _():
    return


@app.cell
def _(mo):
    get_beta_a, set_beta_a = mo.state(2)
    get_beta_b, set_beta_b = mo.state(2)
    return get_beta_a, get_beta_b, set_beta_a, set_beta_b


@app.function
def clamp_n_round(n: float, lo: float, hi: float, to: int = 3) -> float:
    return round(min(hi, max(lo, n)), to)


@app.cell
def _():
    beta_ab_lo = 1
    beta_ab_hi = 21
    return beta_ab_hi, beta_ab_lo


@app.cell
def _(beta_ab_hi, beta_ab_lo, get_beta_a, mo, set_beta_a):
    w_beta_a = mo.ui.slider(
        label=r'$\alpha$',
        start=beta_ab_lo,
        stop=beta_ab_hi,
        step=0.5,
        value=clamp_n_round(get_beta_a(), beta_ab_lo, beta_ab_hi),
        show_value=True,
        on_change=set_beta_a,
        full_width=True,
        debounce=True,
    )
    return (w_beta_a,)


@app.cell
def _(beta_ab_hi, beta_ab_lo, get_beta_b, mo, set_beta_b):
    w_beta_b = mo.ui.slider(
        label=r'$\beta$',
        start=beta_ab_lo,
        stop=beta_ab_hi,
        step=0.5,
        value=clamp_n_round(get_beta_b(), beta_ab_lo, beta_ab_hi),
        show_value=True,
        on_change=set_beta_b,
        full_width=True,
        debounce=True,
    )
    return (w_beta_b,)


@app.cell
def _(fd, get_beta_a, get_beta_b):
    beta = fd.Beta(get_beta_a(), get_beta_b())
    return (beta,)


@app.cell
def _(Callable, set_beta_a, set_beta_b):
    def get_beta_ab_setter(
        fn: Callable[[float, float], tuple[float, float]],
    ) -> Callable[[float, float], None]:
        def setter(x: float, y: float) -> None:
            a, b = fn(x, y)
            set_beta_a(a)
            set_beta_b(b)

        return setter
    return (get_beta_ab_setter,)


@app.cell
def _(beta, fd, get_beta_ab_setter, mo):
    beta_mode_lo = 0.01
    beta_mode_hi = 0.99
    w_beta_mode = mo.ui.slider(
        label='mode',
        start=beta_mode_lo,
        stop=beta_mode_hi,
        step=0.01,
        value=clamp_n_round(beta.mode, beta_mode_lo, beta_mode_hi),
        show_value=True,
        full_width=True,
        on_change=lambda m: get_beta_ab_setter(fd.ab_from_mode_trials)(m, beta.trials),
        debounce=True,
    )
    return (w_beta_mode,)


@app.cell
def _(beta, fd, get_beta_ab_setter, mo):
    beta_trials_lo = 0
    beta_trials_hi = 40
    w_beta_trials = mo.ui.slider(
        label='trials',
        start=beta_trials_lo,
        stop=beta_trials_hi,
        step=1,
        value=clamp_n_round(beta.trials, beta_trials_lo, beta_trials_hi),
        show_value=True,
        full_width=True,
        on_change=lambda t: get_beta_ab_setter(fd.ab_from_mode_trials)(beta.mode, t),
        debounce=True,
    )
    return (w_beta_trials,)


@app.cell
def _(beta, fd, get_beta_ab_setter, mo):
    beta_mu_lo = 0.01
    beta_mu_hi = 0.99
    w_beta_mu = mo.ui.slider(
        label=r'mean $\mu$',
        start=beta_mu_lo,
        stop=beta_mu_hi,
        step=0.01,
        value=clamp_n_round(beta.mu, beta_mu_lo, beta_mu_hi),
        show_value=True,
        full_width=True,
        on_change=lambda mu: get_beta_ab_setter(fd.ab_from_mu_k)(mu, beta.k),
        debounce=True,
    )
    return (w_beta_mu,)


@app.cell
def _(beta, fd, get_beta_ab_setter, mo):
    beta_k_lo = 2
    beta_k_hi = 42
    w_beta_k = mo.ui.slider(
        label='concentration $k$',
        start=beta_k_lo,
        stop=beta_k_hi,
        step=1,
        value=clamp_n_round(beta.k, beta_k_lo, beta_k_hi),
        show_value=True,
        full_width=True,
        on_change=lambda k: get_beta_ab_setter(fd.ab_from_mu_k)(beta.mu, k),
        debounce=True,
    )
    return (w_beta_k,)


@app.cell
def _(beta, fd, get_beta_ab_setter, mo):
    beta_var_lo = 0.001
    beta_var_hi = 0.06
    w_beta_var = mo.ui.slider(
        label='variance',
        start=beta_var_lo,
        stop=beta_var_hi,
        step=0.005,
        value=clamp_n_round(beta.var, beta_var_lo, beta_var_hi, to=5),
        show_value=True,
        full_width=True,
        on_change=lambda var: get_beta_ab_setter(fd.ab_from_mu_var)(beta.mu, var),
        debounce=True,
    )
    return (w_beta_var,)


@app.cell
def _(Literal, alt, fd, ft, pl):
    def plot_beta(
        x: ft.Array,
        beta: fd.Beta,
        kind: Literal['pdf', 'cdf', 'both', 'hboth', 'vboth', 'sep'] = 'pdf',
        width: int = 500,
        height: int = 300,
    ) -> alt.Chart | tuple[alt.Chart, alt.Chart]:
        df = pl.DataFrame({'x': x, 'pdf': beta.pdf(x), 'cdf': beta.cdf(x)})

        hover = alt.selection_point(
            fields=['x'], nearest=True, on='pointerover', empty=False, clear='mouseout'
        )

        base = alt.Chart(df).encode(
            x=alt.X('x:Q', title='x', scale=alt.Scale(domain=[0, 1]))
        )

        # Transparent selectors that capture the hover event across the entire chart area
        selectors = (
            base.mark_point()
            .encode(
                opacity=alt.value(0),
            )
            .add_params(hover)
        )

        # Vertical Rule that appears on both charts at the selected x
        rule = base.mark_rule(color='gray').transform_filter(hover)

        # Set minimum widths and heights
        width = max(100, width)
        height = max(100, height)
        base_props = {'width': width, 'height': height}

        # Common X-axis label that follows the rule (creates a label at the bottom)
        text_x = base.mark_text(align='center', dy=10).encode(
            y=alt.value(base_props['height'] - 20),  # Position at bottom of chart
            text=alt.condition(hover, alt.Text('x:Q', format='.2f'), alt.value(' ')),
        )

        base_title = f'Beta ({beta.a:.2g}, {beta.b:.2g})'

        kind = kind.lower().strip()
        # Points that highlight the exact intersection on the line
        if kind in ('pdf', 'both', 'hboth', 'vboth', 'sep'):
            line_pdf = base.mark_line().encode(y=alt.Y('pdf:Q', title='PDF'))
            point_pdf = line_pdf.mark_point(filled=True).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            )
            # Text labels for X and Y values
            # We offset them slightly so they don't overlap the point
            text_pdf = line_pdf.mark_text(
                align='left', dx=5, dy=-5, color='steelgray'
            ).encode(
                text=alt.condition(
                    hover, alt.Text('pdf:Q', format='.2f'), alt.value(' ')
                ),
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            )
            # We layer: Line -> Selectors -> Rule -> Highlight Point -> Text Label
            chart_pdf = alt.layer(
                line_pdf, selectors, rule, point_pdf, text_pdf, text_x
            ).properties(title=f'{base_title} PDF', **base_props)

        if kind in ('cdf', 'both', 'hboth', 'vboth', 'sep'):
            line_cdf = base.mark_line(color='chocolate').encode(
                y=alt.Y('cdf:Q', title='CDF', scale=alt.Scale(domain=[0, 1])),
            )
            point_cdf = line_cdf.mark_point(filled=True).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            )
            text_cdf = line_cdf.mark_text(
                align='left', dx=5, dy=-5, color='chocolate'
            ).encode(
                text=alt.condition(
                    hover, alt.Text('cdf:Q', format='.2f'), alt.value(' ')
                ),
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            )
            chart_cdf = alt.layer(
                line_cdf, selectors, rule, point_cdf, text_cdf, text_x
            ).properties(title=f'{base_title} CDF', **base_props)

        match kind:
            case 'pdf':
                return chart_pdf
            case 'cdf':
                return chart_cdf
            case 'both' | 'vboth':
                return chart_pdf & chart_cdf
            case 'hboth':
                return chart_pdf | chart_cdf
            case 'sep':
                return chart_pdf, chart_cdf
    return (plot_beta,)


@app.cell
def _(np):
    beta_xax = np.linspace(0, 1, 500)
    return (beta_xax,)


@app.cell
def _(mo):
    w_beta_chart_width = mo.ui.slider(
        label='chart width',
        start=100,
        stop=1920,
        step=10,
        value=500,
        show_value=True,
        full_width=True,
        debounce=True,
    )
    return (w_beta_chart_width,)


@app.cell
def _(mo):
    w_beta_chart_height = mo.ui.slider(
        label='chart height',
        start=100,
        stop=1200,
        step=10,
        value=300,
        show_value=True,
        full_width=True,
        debounce=True,
    )
    return (w_beta_chart_height,)


if __name__ == "__main__":
    app.run()
