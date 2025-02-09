<h1 align='center'>fuz</h1>

**fuz** is a library for probability fusion, allowing the merging of distribution functions in a principled manner.

with **fuz**, you can do things like bayesian evidence updating, optimal bayesian ranking, and fuzzy logic.

## [demos](project:mo/index.md)

allow 30s-1min for the interactive demos to load.

### optimal bayesian ranking

- ch1: bernoulli to beta. [static](project:mo/optimal-bayesian-ranking-ch1-static.md) / [interactive](project:mo/optimal-bayesian-ranking-ch1/index.md)
- ch2: to infinity. [static](project:mo/optimal-bayesian-ranking-ch2-static.md) / [interactive](project:mo/optimal-bayesian-ranking-ch2/index.md)

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

## quickstart

`pip install fuz`


```{toctree}
:maxdepth: 1

demos <mo/index>
api <api/index>
```