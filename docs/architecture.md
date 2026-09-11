# Architecture

```
src/
├── __init__.py            # __version__
├── cli.py                 # probviz app|test|version
├── distributions/
│   ├── base.py            # Distribution ABC: pdf/cdf/ppf/rvs + moments + support
│   ├── continuous.py      # 10 univariate continuous
│   ├── discrete.py        # 6 univariate discrete
│   ├── multivariate.py    # MVN / Dirichlet / MVT / Wishart + biv./simplex plots
│   ├── copulas.py         # Gaussian / Clayton / Gumbel / t + fit_copula_to_data
│   └── mixtures.py        # MixtureDistribution (1-D EM) + sklearn GMM/BGMM + BIC select
├── fitting/               # DistributionFitter / BayesianEstimator / GoodnessOfFit
├── monte_carlo/           # MonteCarloSimulator / VarianceReduction / QMC
├── statistical_tests/     # hypothesis / nonparametric / descriptive (dict-returning)
├── utils/                 # validation / preprocessing / plotting / structured logging
└── visualizers/           # public facade: plot_pdf / plot_cdf / plot_comparison
web/app.py                 # Streamlit UI (16 univariate only, by design)
tests/                     # 688 tests, 11 files + conftest.py
```

## Conventions

- Distributions wrap SciPy frozen distributions (`self._dist`) and add validated
  parameter accessors (`get_parameters` / `set_parameters` /
  `get_parameter_bounds`).
- Sampling uses local RNGs (`np.random.default_rng(seed)`); global
  `np.random.seed` is never mutated by library code.
- Statistical helpers return plain `dict`s for JSON-friendliness.
- Plotting helpers return `(fig, ax)` and never call `plt.show()`.
