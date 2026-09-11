# Probability Distribution Visualizer

An interactive, comprehensive probability distribution visualizer with a Streamlit web
interface, a typed Python API, and toolkits for fitting, Monte Carlo simulation, and
statistical testing.

## What this project contains

**Univariate distributions (16)** — 10 continuous and 6 discrete, all with
`pdf` / `cdf` / `ppf` / `rvs` / statistics:

- Continuous: Normal, Exponential, Uniform, Beta, Gamma, Chi-Square, Student-t,
  Weibull, Lognormal, Cauchy (`src/distributions/continuous.py`)
- Discrete: Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric,
  Discrete Uniform (`src/distributions/discrete.py`)

**Beyond univariate:**

- Multivariate: Multivariate Normal, Dirichlet, Multivariate Student-t, Wishart
  (`src/distributions/multivariate.py`)
- Copulas: Gaussian, Clayton, Gumbel, Student-t (+ `fit_copula_to_data`)
  (`src/distributions/copulas.py`)
- Mixtures: `MixtureDistribution`, sklearn-backed `GaussianMixtureModel`,
  `BayesianGMM`, `select_optimal_components` (`src/distributions/mixtures.py`)
- Distribution fitting & model selection: `DistributionFitter`,
  `BayesianEstimator`, `GoodnessOfFit` (`src/fitting/`)
- Monte Carlo: `MonteCarloSimulator`, `VarianceReduction`,
  `QuasiMonteCarloSimulator` (`src/monte_carlo/`)
- Statistical tests: hypothesis, non-parametric, and descriptive helpers
  (`src/statistical_tests/`)
- Utilities: validation, preprocessing, plotting, structured logging
  (`src/utils/`), plus a `src.visualizers` facade over the plotting helpers
- Web app: Streamlit UI for the 16 univariate distributions (`web/app.py`;
  run with `streamlit run web/app.py` or `probviz app`)

## Repository map

```
.
├── src/
│   ├── distributions/   # base, continuous, discrete, multivariate, copulas, mixtures
│   ├── fitting/         # MLE / Bayesian estimation / goodness-of-fit
│   ├── monte_carlo/     # simulation, variance reduction, QMC
│   ├── statistical_tests/  # hypothesis, nonparametric, descriptive
│   ├── utils/           # validation, preprocessing, plotting, logger
│   ├── visualizers/     # public plotting facade (plot_pdf/plot_cdf/plot_comparison)
│   └── cli.py           # `probviz` command (app/test/version)
├── web/                 # Streamlit app
├── tests/               # 688-test pytest suite (11 files + conftest.py)
├── examples/            # runnable scripts
├── docs/                # MkDocs site (published to GitHub Pages)
└── .github/workflows/   # CI, docs, release
```

See [Architecture](architecture.md) for module responsibilities,
[API Reference](api.md) for the full symbol list, and
[Archive](archive/index.md) for historical development notes.
