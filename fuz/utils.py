"""Miscellaneous utilities."""

from typing import Any


def dl_to_ld(dict_of_lsts: dict[str, list]) -> list[dict[str, Any]]:
    """Convert a dictionary of lists to a list of dictionaries.

    Modified from:
    https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    """
    return [
        dict(zip(dict_of_lsts.keys(), tup, strict=True))
        for tup in zip(*dict_of_lsts.values(), strict=True)
    ]


def ld_to_dl(lst_of_dicts: list[dict[str, Any]]) -> dict[str, list]:
    """Convert a list of dictionaries to a dictionary of lists.

    Modified from:
    https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    """
    return {k: [d[k] for d in lst_of_dicts] for k in lst_of_dicts[0]}

def support_autoreload() -> None:
    """Plum-dispatch iPython autoreload support."""
    from plum import activate_autoreload

    activate_autoreload()
