"""Ranking functions."""
import numpy as np

def bayes_avg(mu_item: float, mu_all: float, n_item: float, n_all: float) -> float:
    weight = n_item / (n_all + n_item)
    return weight * mu_item + (1 - weight) * mu_all

def ros(ratings: np.ndarray, counts: np.ndarray | None=None):
    """Classic rule of succession."""
    if counts is None:
        counts = np.ones_like(ratings)
    return (counts*ratings + 1) / (counts + 2)

def inf_weight(ratings: np.ndarray, counts: np.ndarray | None=None, midpoint: float=0.5):
    """Miraia's infinite dimension weighting, stably extending the rule of succession."""
    if counts is None:
        return np.sum((np.asarray(ratings) - midpoint) * np.ones_like(ratings))
    return np.sum((np.asarray(ratings) - midpoint) * np.asarray(counts))


def inf_ros(ratings: np.ndarray, counts: np.ndarray | None=None, dims: float=2, midpoint: float=0.5):
    """Rule of succession, using Miraia's infinite dimension weight extension.
    
    beta equivalent: dims = 2
    dirichlet: dims = x

    See Also
    --------
    inf_weight
    """
    if counts is None:
        n = np.asarray(ratings).size # assuming 1 count per rating
    else:
        n = np.sum(counts)
    return inf_weight(ratings, counts=counts, midpoint=midpoint) / (n+dims) + midpoint