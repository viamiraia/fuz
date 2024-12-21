"""Reusable helper functions for marimo demos."""

from collections.abc import Sequence

import marimo as mo

from fuz.rank import rating_to_moons


def make_star_widget(n_stars: int = 5, star0: Sequence[int] | int = 1, max_ratings: int = 25):
    if isinstance(star0, int):
        star0 = [star0] * n_stars
    w_stars, stacks = [], []
    for i in range(n_stars):
        w_star = mo.ui.slider(0, max_ratings, 1, star0[i], full_width=True, show_value=True)
        star_str = f'{i+1} {rating_to_moons(i, max_rating=n_stars)}'
        stack = mo.hstack([mo.md(star_str), w_star], widths=[1, 3], gap=0, align='center')
        w_stars.append(w_star)
        stacks.append(stack)
    layout = mo.vstack(
        [
            mo.hstack(
                [mo.md('stars'), mo.md('no. of ratings')], widths=[1, 3], gap=0, align='center'
            ),
            *stacks,
        ],
        gap=0,
    )
    return layout, w_stars
