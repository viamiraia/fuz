from functools import cached_property
from typing import Any

import numpy as np
from attrs import field, frozen
from scipy.stats import dirichlet

import fuz.types as ft


@frozen
class Dirichlet:
    alpha: ft.ArrVec = field(converter=np.array)

    @cached_property
    def k(self):
        return len(self.alpha)

    @cached_property
    def a0(self):
        return self.alpha.sum()

    @cached_property
    def d(self):
        return dirichlet(self.alpha)

    @cached_property
    def mean(self):
        return self.d.mean()

    @cached_property
    def var(self):
        return self.d.var()

    @cached_property
    def cov(self):
        return self.d.cov()

    @cached_property
    def entropy(self):
        return self.d.entropy()

    @cached_property
    def mode(self):
        """The mode of the distribution."""
        return (self.alpha - 1) / (self.a0 - self.k)

    def __getattr__(self, attr: str) -> Any:
        """Forward any unknown attributes to the underlying scipy distribution."""
        return getattr(self.d, attr)


@frozen
class Scored(Dirichlet):
    scores: ft.ArrVec

    @cached_property
    def mu(self):
        return np.sum(self.scores * self.mean)

    @cached_property
    def mo(self):
        return np.sum(self.scores * self.mode)
    
    
