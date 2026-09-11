# Multivariate, Copulas & Mixtures

## Multivariate (`src.distributions.multivariate`)

- `MultivariateNormalDistribution(mean, cov)` — full pdf/logpdf/rvs/mean/cov;
  `plot_bivariate_normal(dist)` gives contour + 3-D surface for d=2.
- `DirichletDistribution(alpha)` — simplex sampling; `plot_dirichlet_simplex`
  for d=3.
- `MultivariateStudentT(df, loc, shape)` — pdf via the closed-form density,
  sampling via the normal/χ² representation with a local `default_rng`.
- `WishartDistribution(df, scale)` — distribution over PSD matrices with
  mean/mode/pdf/logpdf/rvs.

## Copulas (`src.distributions.copulas`)

| Copula | Status |
|---|---|
| Gaussian | cdf/pdf/rvs/kendall_tau (any dimension) |
| Clayton | cdf (any d); **pdf/rvs currently bivariate-only** |
| Gumbel | cdf (any d); **pdf/rvs currently bivariate-only** |
| Student-t | rvs/kendall_tau; **cdf/pdf intentionally not implemented** (requires multivariate-t integration — use Monte Carlo via `rvs`) |

`fit_copula_to_data(data, copula_type, method)` fits Gaussian/Clayton/Gumbel/t
from pseudo-observations. It validates Kendall's τ ranges and documents the
`t` degrees-of-freedom simplification (`df=4`).

```python
from src.distributions import GaussianCopula
import numpy as np

cop = GaussianCopula(np.array([[1.0, 0.6], [0.6, 1.0]]))
u = cop.rvs(size=1000, random_state=42)   # uniform margins with Gaussian dependence
```

## Mixtures (`src.distributions.mixtures`)

- `MixtureDistribution(components, weights)` — pdf/cdf/rvs/mean/var + `fit_em`
  (EM for 1-D Gaussian mixtures; guards empty data, `n_components > n`, zero
  responsibilities, and zero variances; convergence is checked **after** the
  M-step so returned parameters are never stale).
- `GaussianMixtureModel` / `BayesianGMM` — sklearn-backed fitting, predict,
  BIC/AIC, active-component counts.
- `select_optimal_components(data, max_components)` — BIC sweep.

!!! warning "Scope"
    `MixtureDistribution` assumes 1-D components. Multivariate mixtures should
    use `GaussianMixtureModel`/`BayesianGMM`.
