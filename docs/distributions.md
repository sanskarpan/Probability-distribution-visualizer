# Univariate Distributions

All 16 univariate distributions share the `Distribution` base-class API
(`src/distributions/base.py`):

`pdf(x)` · `cdf(x)` · `ppf(q)` · `rvs(size, random_state)` · `mean()` · `var()` ·
`std()` · `median()` · `mode()` · `skewness()` · `kurtosis()` · `entropy()` ·
`interval(alpha)` · `get_statistics()` · `support()` · parameter getters/setters
with validated bounds (`get_parameter_bounds()`).

## Continuous (10)

| Distribution | Parameters | Support | Typical use |
|---|---|---|---|
| Normal | μ, σ | (−∞, ∞) | measurement error, CLT |
| Exponential | λ (rate) | [0, ∞) | inter-arrival times |
| Uniform | a, b | [a, b] | bounded ignorance prior |
| Beta | α, β | [0, 1] | proportions, Bayesian rates |
| Gamma | shape k, scale θ | [0, ∞) | waiting times, claims |
| Chi-Square | df | [0, ∞) | variance inference |
| Student-t | df | (−∞, ∞) | heavy tails, small samples |
| Weibull | shape, scale | [0, ∞) | reliability, lifetimes |
| Lognormal | μ, σ | (0, ∞) | incomes, sizes |
| Cauchy | x₀, γ | (−∞, ∞) | heavy-tail counterexample (mean/variance undefined → `nan`) |

## Discrete (6)

| Distribution | Parameters | Support | Typical use |
|---|---|---|---|
| Binomial | n, p | {0..n} | successes in n trials |
| Poisson | λ | {0, 1, …} | counts, arrivals |
| Geometric | p | {1, 2, …} | trials until first success |
| Negative Binomial | n, p | {n, n+1, …} | trials until n-th success |
| Hypergeometric | M, n, N | bounded | sampling without replacement |
| Discrete Uniform | a, b | {a..b} | fair dice |

## Recipes

```python
from src.distributions import NormalDistribution, PoissonDistribution

n = NormalDistribution(mu=0, sigma=1)
print(n.interval(0.95))      # 95% central interval
print(n.ppf(0.975))          # quantile
print(n.get_statistics())    # everything at once

pois = PoissonDistribution(lambda_param=3.0)
print(pois.pmf if hasattr(pois, "pmf") else pois.pdf([0, 1, 2, 3]))
```

!!! note "Cauchy moments"
    Skewness/kurtosis/mean/variance are undefined for Cauchy; the API surfaces
    `nan` from SciPy rather than masking it. `get_statistics()` includes the
    `nan` so downstream code can decide how to handle it.

!!! note "Mode search"
    `mode()` falls back to a bounded grid search when SciPy has no closed form.
    For heavy-tailed or infinite-support discretes the grid is intentionally
    bounded; treat `mode()` as an approximation there.
