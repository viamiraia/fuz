import marimo

__generated_with = '0.17.7'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from scipy.integrate import trapezoid

    from fuz import lint

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1 align='center'>fuz-lint</h1>
    <h3 align='center'>chapter 1: integration in log space</h3>
    <p align='center'>by miraia s. chiou © 2025</p>
    """)
    return


@app.cell
def _():
    from fuz.dists import Beta

    return (Beta,)


@app.cell
def _(Beta):
    Beta(9, 4)
    return


if __name__ == '__main__':
    app.run()
