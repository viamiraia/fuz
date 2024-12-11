"""Conversion functions."""

import math
from typing import Any

import numpy as np

import fuz.types as ft


def dl_to_ld(dict_of_lsts: ft.DictOfLists) -> ft.ListOfDicts:
    """Convert a dictionary of lists to a list of dictionaries.

    Modified from:
    https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    """
    return [
        dict(zip(dict_of_lsts.keys(), tup, strict=True))
        for tup in zip(*dict_of_lsts.values(), strict=True)
    ]


def ld_to_dl(lst_of_dicts: ft.ListOfDicts) -> ft.DictOfLists:
    """Convert a list of dictionaries to a dictionary of lists.

    Modified from:
    https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    """
    return {k: [d[k] for d in lst_of_dicts] for k in lst_of_dicts[0]}


def rating_to_moons(r: ft.Scalar, max_rating: int = 5) -> str:
    """Convert a numeric rating to a visual representation with moon emojis.

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
    remain = int(np.round((r % 1) * 4))
    moon_str = '🌕' * full_moons
    if r < max_rating:
        moon_str += moon_map[remain]
    return moon_str.ljust(max_rating, '🌑')


def prop_to_stars(r: float | np.ndarray) -> np.ndarray:
    """Linearly convert from a 0-1 to a 1-5 scale."""
    return np.interp(r, (0, 1), (1, 5))


def stars_to_prop(s: float | np.ndarray) -> np.ndarray:
    """Linearly convert from a 1-5 to a 0-1 scale."""
    return np.interp(s, (1, 5), (0, 1))


def prop_to_moons(r: float | np.ndarray) -> str | np.ndarray:
    """Convert 1 or more ratings to visual representations with moon emojis."""
    if not isinstance(r, np.ndarray):
        return rating_to_moons(float(prop_to_stars(r)))
    return np.vectorize(rating_to_moons)(prop_to_stars(r))
