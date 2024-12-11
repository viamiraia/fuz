"""Ranking functions."""


def bayes_avg(mu_item: float, mu_all: float, n_item: float, n_all: float) -> float:
    weight = n_item / (n_all + n_item)
    return weight * mu_item + (1 - weight) * mu_all
