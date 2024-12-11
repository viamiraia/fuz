"""Miscellaneous utilities."""


def support_autoreload() -> None:
    """Plum-dispatch iPython autoreload support."""
    from plum import activate_autoreload

    activate_autoreload()
