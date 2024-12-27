<!-- begin-short -->

<h1 align='center'>fuz</h1>

**fuz** is a library for probability fusion, allowing the merging of distribution functions in a principled manner.

with **fuz**, you can do things like bayesian evidence updating, optimal bayesian ranking, and fuzzy logic.

## highlights

- pooling
  - linear, multiplicative, geometric, holder ...etc.
- fuzzy logic extension to distribution functions
- numerical integration in logarithmic space for stability
- `NaN` and logarithmic complex number handling
- distribution tools

### upcoming features

- wider compatibility with `narwhals`
  - dataframes, functions, samples ...etc.
- initial gpu support with `jax`
- more documentation 😹

## demos

check out the marimo demos at:

- ch1. [bernoulli to beta](https://code.raia.fun/fuz/mo/optimal-bayesian-ranking-ch1/index.html)
- ch2. [to infinity](https://code.raia.fun/fuz/mo/optimal-bayesian-ranking-ch2/index.html)

## quickstart

`pip install fuz`

### distributions

```python
import fuz.dists as fd
a = fd.Beta(5,4)
a.stats
```

```python
b = fd.beta_from_mode_trials(0.9,10)
d = fd.Dirichlet([3,6,2])
```

### logarithmic manipulation

```python
import numpy as np
from fuz.log import complex_lsub, lsimp_irreg

a = np.log([[1,np.nan],[3,4]])
b = np.log([[3,2],[6,1]])
print(complex_lsub(a,b))

x = np.linspace(0,1,1025)
b = fd.beta_from_mu_k(0.3,9)
c = fd.beta_from_mode_k(0.3,9)
y = b.logpdf(x) * c.logcdf(x)
np.exp(lsimp_irreg(y,x))
```
note - integrating `y` here gives you the win rate of `b` over `c`. see the demo folder for more.

### fuzzy logic

```python
import fuz.logic as fzl

x = np.linspace(0,1,1025)
b = fd.beta_from_mode_var(0.7,0.01)
c = fd.beta_from_mu_var(0.7,0.01)
b_negpdf = fzl.negf(b.pdf)
negb_and_c = fzl.ands(x, b_negpdf(x), c.pdf(x))
```
### pooling

### ranking

### more

see demo folder (`pip install marimo` first) for more usage.

## other

the official pronunciation of **fuz** is a fugued function of fuzzy fusion 😉

### abstract

**fuz** is a python library for fusion of probability distributions, bayesian evidence updating and ranking, fuzzy logic, operations in logarithmic space, and more.
**fuz** includes several original contributions, including a performant bayesian averaging/ranking algorithm that is mathematically optimal, equations for determining whether one pdf "wins" over another, and
multiple demonstrations of equivalence. these include the equivalence of simple kalman filtering, bayesian updating, and multiplicative (upco) pooling, the equivalence of
mode-parameterization of beta and dirichlet distributions to optimal bayesian averaging, useful limits as the dirichlet $\alpha_0$ approaches infinity, and various equivalences for
discrete distributions.

### acknowledgements

thanks to jane liou for her useful insights and support!
<!-- end-short -->

## 😿 please help 😽

contributions are extremely welcome! i'm especially looking for:

- co-maintainers
- development help
- mathematician/computer scientist help
- real-world feedback
- potential help with an academic paper, if this qualifies

between life, work, and wayyy too many hobbies, research directions, and passion projects, i am always really short on time. also, i'm honestly not that good at math and cs, and there's definitely ways to improve that i haven't thought of. any possible help is sincerely appreciated. this is also my first publicly published python package😸, so please let me know if there's anything i can do differently.

you can also support my original research by donating at [ko-fi](https://ko-fi.com/viamiraia) and/or following me @viamiraia on most social media (@raia.fun on bluesky)
