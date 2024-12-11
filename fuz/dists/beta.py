"""Anything related to the beta distribution."""

import math
from collections.abc import Callable
from functools import cached_property, partial
from typing import Any, cast

import numpy as np
import scipy.stats as st
from attrs import frozen
from scipy.optimize import newton_krylov
from scipy.stats._distn_infrastructure import rv_continuous_frozen

import fuz.types as ft

# %% Frozen Attrs Beta Distribution Class


@frozen
class Beta:
    """An immutable beta distribution with parameters a and b, and optionally loc and scale.

    Parameters
    ----------
    a
        The first parameter of the beta distribution.
    b
        The second parameter of the beta distribution.
    loc
        The location parameter of the distribution. Defaults to 0.
    scale
        The scale parameter of the distribution. Defaults to 1.

    Attributes
    ----------
    is_std
        Whether the distribution has standard location and scale (0 and 1).
    k
        The parameter k = a + b of the distribution.
    trials
        The effective number of trials, k - 2.
    heads
        The number of heads if the distribution represented coin flips, a - 1.
    tails
        The number of tails if the distribution represented coin flips, b - 1.
    mu
        The mean of the distribution.
    var
        The variance of the distribution.
    sd
        The standard deviation of the distribution.
    mode
        The mode of the distribution.
    d
        The corresponding scipy distribution.
    entropy
        The entropy of the distribution.
    skew
        The skewness of the distribution.
    kurtosis
        The kurtosis of the distribution.
    median
        The median of the distribution.
    stats
        A dictionary of summary statistics.

    Notes
    -----
    Any unknown attributes are forwarded to the underlying scipy distribution.
    """

    a: ft.Scalar
    b: ft.Scalar
    loc: ft.Scalar = 0
    scale: ft.Scalar = 1

    def _reloc(self, x: ft.Broadcast) -> ft.Broadcast:
        """Recenter and rescale a value according to the distribution loc and scale."""
        return self.loc + x * self.scale

    @cached_property
    def is_std(self) -> bool:
        """Check if distribution has standard location and scale (0 and 1)."""
        return self.loc == 0 and self.scale == 1  # pyright: ignore[reportReturnType]

    @cached_property
    def k(self) -> ft.Scalar:
        """Get distribution parameter k (a+b)."""
        return self.a + self.b

    @cached_property
    def trials(self) -> ft.Scalar:
        """Get effective # of trials (k-2)."""
        return self.k - 2

    @cached_property
    def heads(self) -> ft.Scalar:
        """Get the number of heads if the distribution represented coin flips (a-1)."""
        return self.a - 1

    @cached_property
    def tails(self) -> ft.Scalar:
        """Get the number of tails if the distribution represented coin flips (b-1)."""
        return self.b - 1

    @cached_property
    def mu(self) -> ft.Scalar:
        """Get the mean of the distribution."""
        return self._reloc(self.a / (self.k))

    @cached_property
    def var(self) -> ft.Scalar:
        """Get the variance of the distribution."""
        return (self.a * self.b) / (self.k**2 * (self.k + 1)) * self.scale**2

    @cached_property
    def sd(self) -> ft.Scalar:
        """Get the standard deviation of the distribution."""
        return math.sqrt(self.var)

    @cached_property
    def mode(self) -> ft.Scalar:
        """Get the mode of the distribution."""
        if math.isclose(self.a, 1) and math.isclose(self.b, 1):
            return self._reloc(0.5)
        if math.isclose(self.k, 2):
            return self._reloc(1) if self.a > self.b else self._reloc(0)
        return self._reloc(self.heads / self.trials)

    @cached_property
    def d(self) -> rv_continuous_frozen:
        """Get the corresponding scipy distribution."""
        return cast(rv_continuous_frozen, st.beta(self.a, self.b, self.loc, self.scale))

    @cached_property
    def entropy(self) -> ft.Scalar:
        """Get the entropy of the distribution."""
        return self.d.entropy()

    @cached_property
    def skew(self) -> ft.Scalar:
        """Get the skewness of the distribution."""
        return self.d.stats(moments='s')

    @cached_property
    def kurtosis(self) -> ft.Scalar:
        """Get the kurtosis of the distribution."""
        return self.d.stats(moments='k')

    @cached_property
    def median(self) -> ft.Scalar:
        """Get the median of the distribution."""
        return self.d.median()

    @cached_property
    def stats(self) -> dict[str, ft.Scalar]:
        """Get summary statistics of the distribution."""
        return {
            'a': self.a,
            'b': self.b,
            'k': self.k,
            'trials': self.trials,
            'mode': self.mode,
            'mean': self.mu,
            'median': self.median,
            'variance': self.var,
            'standard deviation': self.sd,
            'skewness': self.skew,
            'kurtosis': self.kurtosis,
            'entropy': self.entropy,
        }

    def __getattr__(self, attr: str) -> Any:
        """Forward any unknown attributes to the underlying scipy distribution."""
        return getattr(self.d, attr)


# %% Alpha and Beta Shape Parameter Functions


def ab_from_mode_trials(m: ft.Broadcast, trials: ft.Broadcast) -> ft.ABBroadcast:
    """Calculate beta distribution from mode and trials = k - 2.

    Parameters
    ----------
    m
        desired mode of the beta distribution.
    trials
        desired "number of trials" of the beta distribution, in a Bayesian context.

    Returns
    -------
    rt.AlphaBeta
        the solved alpha and beta shape parameters.

    """
    a = m * (trials) + 1
    return a, trials + 2 - a


def ab_from_mode_k(m: ft.Broadcast, k: ft.Broadcast) -> ft.ABBroadcast:
    """Calculate beta distribution from mode and k = a + b.

    See https://doingbayesiandataanalysis.blogspot.com/2012/06/beta-distribution-parameterized-by-mode.html

    Parameters
    ----------
    m
        desired mode of the beta distribution.
    k
        desired "number of trials" + 2 of the beta distribution, in a Bayesian context.

    Returns
    -------
    rt.AlphaBeta
        the solved alpha and beta shape parameters.

    """
    a = m * (k - 2) + 1
    # b = (1 - m) * (k - 2) + 1
    return a, k - a


def ab_from_mu_k(mu: ft.Broadcast, k: ft.Broadcast) -> ft.ABBroadcast:
    """Get beta distribution parameters from the mean mu and shape k.

    Parameters
    ----------
    mu
        The mean of the distribution.
    k
        The shape parameter of the distribution.

    Returns
    -------
    rt.AlphaBeta
        the solved alpha and beta shape parameters.
    """
    a = mu * k
    return a, k - a


def ab_from_mu_var(mu: ft.Broadcast, var: ft.Broadcast) -> ft.ABBroadcast:
    """Get beta distribution parameters from the mean mu and variance var.

    See https://statproofbook.github.io/P/beta-mome.html

    Parameters
    ----------
    mu
        the mean of the distribution.
    var
        the variance of the distribution.

    Returns
    -------
    rt.AlphaBeta
        the solved alpha and beta shape parameters.
    """
    common = (mu * (1 - mu) / var) - 1
    return mu * common, (1 - mu) * common


def ab_from_mode_var(m: ft.Broadcast, var: ft.Broadcast) -> ft.ABBroadcast:
    """Get beta distribution parameters from the mode and variance.

    Solves using Krylov approximation method.
    newton_krylov is the only one of the solvers that returns NoConvergence.
    Ex. fsolve returns the wrong answer when it should have NoConvergence.

    Parameters
    ----------
    m
        the mode of the distribution.
    var
        the variance of the distribution.

    Returns
    -------
    rt.AlphaBeta
        the solved alpha and beta shape parameters.
    """

    def func_to_solve(x: ft.ABTuple) -> ft.ABTuple:
        a, b = x
        m_expr = (a - 1) / (a + b - 2) - m
        v_expr = (a * b) / ((a + b + 1) * (a + b) ** 2) - var
        return m_expr, v_expr

    ab = newton_krylov(func_to_solve, [2, 2])
    return ab[0], ab[1]


# %% Beta Distribution Convenience Functions


def beta_from(ab_fn: Callable[..., ft.ABTuple], *args: Any, **kwargs: Any) -> Beta:
    """Create a Beta distribution using a function that returns alpha and beta parameters.

    Parameters
    ----------
    ab_fn
        A function that calculates and returns the alpha and beta parameters of the distribution.
    *args
        Positional arguments to pass to `ab_fn`.
    **kwargs
        Keyword arguments to pass to `ab_fn`.

    Returns
    -------
    Beta
        A Beta distribution object initialized with alpha and beta parameters.

    Notes
    -----
    If `ab_fn` returns numpy arrays for alpha or beta, the first items in the array are
    converted to floats and used to create the beta distribution.
    """
    a, b = ab_fn(*args, **kwargs)
    # Convert numpy array output to floats if necessary
    if isinstance(a, np.ndarray):
        a = float(a.item(0))
    if isinstance(b, np.ndarray):
        b = float(b.item(0))
    # Return Beta distribution with calculated parameters
    return Beta(a, b)


# %%% Make partial functions and their docstrings

_beta_fn_doc_footer = """

Returns
-------
Beta
    Frozen beta distribution attrs object initialized with alpha and beta parameters.
"""

beta_from_mode_trials = partial(beta_from, ab_from_mode_trials)
beta_from_mode_trials.__doc__ = (
    """Make a beta distribution using mode and trials = k - 2.

Parameters
----------
m : rt.Realish
    desired mode of the beta distribution.
trials : rt.Realish
    desired "number of trials" of the beta distribution, in a Bayesian context.
"""
    + _beta_fn_doc_footer
)


beta_from_mode_k = partial(beta_from, ab_from_mode_k)
beta_from_mode_k.__doc__ = (
    """Make a beta distribution using mode and k = a + b.

See https://doingbayesiandataanalysis.blogspot.com/2012/06/beta-distribution-parameterized-by-mode.html

Parameters
----------
m : rt.Realish
    desired mode of the beta distribution.
k : rt.Realish
    desired "number of trials" + 2 of the beta distribution, in a Bayesian context.
"""
    + _beta_fn_doc_footer
)


beta_from_mu_k = partial(beta_from, ab_from_mu_k)
beta_from_mu_k.__doc__ = (
    """Make a beta distribution from the mean mu and shape k.

Parameters
----------
mu : rt.Realish
    The mean of the distribution.
k : rt.Realish
    The shape parameter of the distribution.
"""
    + _beta_fn_doc_footer
)

beta_from_mu_var = partial(beta_from, ab_from_mu_var)
beta_from_mu_var.__doc__ = (
    """Make a beta distribution from the mean mu and variance var.

See https://statproofbook.github.io/P/beta-mome.html

Parameters
----------
mu: rt.Realish
    the mean of the distribution.
var: rt.Realish
    the variance of the distribution.
"""
    + _beta_fn_doc_footer
)

beta_from_mode_var = partial(beta_from, ab_from_mode_var)
beta_from_mode_var.__doc__ = (
    """Make a beta distribution from the mode and variance.

Solves using Krylov approximation method.
newton_krylov is the only one of the solvers that returns NoConvergence.
Ex. fsolve returns the wrong answer when it should have NoConvergence.

Parameters
----------
m: rt.Realish
    the mode of the distribution.
var: rt.Realish
    the variance of the distribution.
"""
    + _beta_fn_doc_footer
)


# %% Prior and Posterior
def get_posterior(prior1: Beta, prior2: Beta) -> Beta:
    return Beta(prior1.a + prior2.a - 1, prior1.b + prior2.b - 1)
