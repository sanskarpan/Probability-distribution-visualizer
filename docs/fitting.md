# Fitting & Model Selection (`src.fitting`)

- `DistributionFitter(data)` — `fit_all()` / `fit_distribution(name)` across
  Normal, Exponential, Gamma (MLE + method-of-moments), Beta, Weibull,
  Lognormal; plus `qq_plot_data` and `calculate_residuals`.
- `BayesianEstimator(data)` — conjugate posteriors for Normal mean/variance,
  Poisson rate, Bernoulli p.
- `GoodnessOfFit` — static chi-square, Kolmogorov–Smirnov, Anderson–Darling,
  Shapiro–Wilk, and Jarque–Bera tests returning plain dicts.

```python
from probviz.fitting import DistributionFitter
import numpy as np

rng = np.random.default_rng(0)
data = rng.normal(0, 1, size=500)
fitter = DistributionFitter(data)
print(fitter.fit_all())   # ranked by goodness-of-fit
```
