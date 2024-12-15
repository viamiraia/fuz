<h1 align='center'>fuz</h1>

**fuz** is a library for probability fusion, allowing the merging of distribution functions in a principled manner.

with **fuz**, you can do things like bayesian evidence updating, optimal bayesian ranking, and fuzzy logic.


:::{admonition} look at these [demos](project:mo/index.md)!
:class: note
optimal bayesian ranking
- ch1. [bernoulli to beta](project:mo/obr1/index.md)
- ch2. [comparisons](project:mo/obr2/index.md)
:::


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


## other

[demos](project:mo/index.md)

```{toctree}
:maxdepth: 1
[demos](project:mo/index.md)
```