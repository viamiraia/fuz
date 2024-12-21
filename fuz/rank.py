"""Ranking functions."""

import numpy as np


def bayes_avg(mu_item: float, mu_all: float, n_item: float, n_all: float) -> float:
    weight = n_item / (n_all + n_item)
    return weight * mu_item + (1 - weight) * mu_all


def ros(ratings: np.ndarray, counts: np.ndarray | None = None):
    """Classic rule of succession."""
    if counts is None:
        counts = np.ones_like(ratings)
    return (counts * ratings + 1) / (counts + 2)


def inf_weight(ratings: np.ndarray, counts: np.ndarray | None = None, midpoint: float = 0.5):
    """Miraia's infinite dimension weighting, stably extending the rule of succession."""
    if counts is None:
        return np.sum((np.asarray(ratings) - midpoint) * np.ones_like(ratings))
    return np.sum((np.asarray(ratings) - midpoint) * np.asarray(counts))


def inf_ros(
    ratings: np.ndarray,
    counts: np.ndarray | None = None,
    dims: float = 2,
    midpoint: float = 0.5,
):
    """Rule of succession, using Miraia's infinite dimension weight extension.

    beta equivalent: dims = 2
    dirichlet: dims = x

    See Also
    --------
    inf_weight
    """
    if counts is None:
        n = np.asarray(ratings).size  # assuming 1 count per rating
    else:
        n = np.sum(counts)
    return inf_weight(ratings, counts=counts, midpoint=midpoint) / (n + dims) + midpoint


import math

import numpy as np


def rating_to_moons(r: float, max_rating: int = 5) -> str:
    """
    Convert a numeric rating to a visual representation with moon emojis.

    Parameters
    ----------
    r
        The rating to convert.
    max_rating
        The maximum rating value. Default is 5.

    Returns
    -------
    str
        The visual representation of the rating as a string of moon emojis.

    Raises
    ------
    ValueError
        If `max_rating` is less than 1 or if `r` is not between 0 and `max_rating`.

    Examples
    --------
    >>> rating_to_moons(4.5)
    '🌕🌕🌕🌕🌗'
    >>> rating_to_moons(2.3, max_rating=10)
    '🌕🌕🌘🌑🌑🌑🌑🌑🌑🌑'
    """
    if max_rating < 1:
        msg = 'max_rating must be at least 1'
        raise ValueError(msg)
    if not 0 <= r <= max_rating:
        msg = f'Rating must be between 0 and {max_rating}'
        raise ValueError(msg)
    moon_map = ('🌑', '🌘', '🌗', '🌖', '🌕')
    full_moons = math.floor(r)
    remain = round((r % 1) * 4)
    moon_str = '🌕' * full_moons
    if r < max_rating:
        moon_str += moon_map[remain]
    return moon_str.ljust(max_rating, '🌑')
