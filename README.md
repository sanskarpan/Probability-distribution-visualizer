# Probability Distribution Visualizer

An interactive probability distribution visualizer with a Streamlit web interface,
a typed Python API, and toolkits for fitting, Monte Carlo simulation, and
statistical testing.

<p align="center">
  <a href="https://github.com/sanskarpan/probviz/actions/workflows/ci.yml"><img src="https://github.com/sanskarpan/probviz/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/sanskarpan/probviz/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-688%20passed-brightgreen" alt="Tests"></a>
  <a href="https://sanskarpan.github.io/probviz/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
  <a href="https://github.com/sanskarpan/probviz/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?logo=streamlit" alt="Streamlit"></a>
  <a href="https://github.com/sanskarpan/probviz/pkgs/container/probviz"><img src="https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker" alt="Docker"></a>
  <a href="https://github.com/sanskarpan/probviz/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome"></a>
</p>

- **Docs (GitHub Pages):** <https://sanskarpan.github.io/probviz/>
- **16 univariate distributions** (10 continuous + 6 discrete) with PDF/PMF, CDF,
  quantiles, sampling, and full statistics — in the web app and the API.
- **Advanced modules** (Python API): multivariate distributions, copulas,
  mixtures/GMM, distribution fitting, Monte Carlo, and statistical tests.
- **Production-ready:** 688-test suite, typed packaging (`pyproject.toml`),
  `probviz` CLI, Docker/Compose, CI with coverage gate, docs + PyPI + Docker
  release pipelines. See [`CHANGELOG.md`](CHANGELOG.md).

## Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.4/docs/assets/normal_sigma_morph.gif" width="250" alt="Normal PDF morphing with sigma">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.4/docs/assets/clt_convergence.gif" width="250" alt="Central Limit Theorem convergence">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.4/docs/assets/copula_dependence.gif" width="210" alt="Gaussian copula dependence sweep">
  <br>
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.4/docs/assets/beta_shape_morph.gif" width="250" alt="Beta PDF shape morph">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.4/docs/assets/mixture_separation.gif" width="250" alt="Gaussian mixture separation">
</p>

*Top: Normal σ sweep · CLT convergence · Gaussian-copula ρ sweep. Bottom: Beta shape sweep · mixture separation. Regenerate with `python examples/generate_media.py`.*

## Quick start

```bash
pip install probviz
probviz app
```

Or from source:

```bash
git clone https://github.com/sanskarpan/probviz.git
cd probviz
pip install -r requirements.txt
streamlit run web/app.py
```

Open `http://localhost:8501`. Full guide: [`QUICKSTART.md`](QUICKSTART.md) ·
[docs quickstart](https://sanskarpan.github.io/probviz/quickstart/).

### Install as a package

```bash
pip install -e .
probviz app        # launch the UI
probviz test       # run tests
probviz version    # print version
```

### Docker

```bash
docker build -t probviz .
docker run -p 8501:8501 probviz
# or
docker compose up --build
```

## What's inside

| Area | Contents |
|---|---|
| Univariate | Normal, Exponential, Uniform, Beta, Gamma, Chi-Square, Student-t, Weibull, Lognormal, Cauchy · Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric, Discrete Uniform |
| Multivariate | Multivariate Normal, Dirichlet, Multivariate Student-t, Wishart |
| Copulas | Gaussian, Clayton, Gumbel, Student-t + `fit_copula_to_data` |
| Mixtures | `MixtureDistribution` (1-D EM), `GaussianMixtureModel`, `BayesianGMM`, BIC selection |
| Fitting | `DistributionFitter`, `BayesianEstimator`, `GoodnessOfFit` |
| Monte Carlo | `MonteCarloSimulator`, `VarianceReduction`, `QuasiMonteCarloSimulator` |
| Tests | hypothesis / nonparametric / descriptive dict-returning helpers |
| Utils | validation, preprocessing, plotting, structured logging; `src.visualizers` facade |

Project layout and conventions: [`docs/architecture.md`](docs/architecture.md).
API reference: [`docs/api.md`](docs/api.md) (rendered on the docs site).

## Python API

```python
import numpy as np
from src.distributions import NormalDistribution, BinomialDistribution

normal = NormalDistribution(mu=0, sigma=1)
x = np.linspace(-4, 4, 200)
pdf, cdf = normal.pdf(x), normal.cdf(x)
samples = normal.rvs(size=1000, random_state=42)
print(normal.get_statistics())
print(normal.interval(0.95), normal.ppf(0.975))

binomial = BinomialDistribution(n=10, p=0.3)
print(binomial.pdf(np.arange(0, 11)))
```

```python
from src.fitting import DistributionFitter
from src.monte_carlo import MonteCarloSimulator

fitter = DistributionFitter(samples)
print(fitter.fit_all())

rng = np.random.default_rng(42)
sim = MonteCarloSimulator(random_seed=42)
res = sim.estimate_probability(lambda: rng.normal(0, 1) > 1.0, num_samples=100_000)
print(res["probability"], res["confidence_interval"])
```

## Known limitations (by design)

- The Streamlit app covers the **16 univariate distributions only**; advanced
  modules are Python-API only.
- `Clayton`/`Gumbel` copula `pdf`/`rvs` are **bivariate-only** (explicit error otherwise).
- `StudentTCopula` has **no closed-form `cdf`/`pdf`**; use Monte Carlo via `rvs`.
- `MixtureDistribution` EM assumes **1-D** components; use the sklearn GMM wrappers
  for multivariate mixtures.
- Cauchy moments are undefined — the API surfaces `nan` instead of masking it.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -q                                  # 688 tests
pytest tests/ -q --cov=src --cov-report=term      # with coverage (gate: 80%)
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --statistics
mypy --config-file=pyproject.toml src/
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (setup, style, tests, PR checklist),
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and [`SECURITY.md`](SECURITY.md).

## License

MIT — see [`LICENSE`](LICENSE). If you use this in research or teaching:

```bibtex
@software{probability_distribution_visualizer,
  title  = {Probability Distribution Visualizer},
  author = {sanskarpan},
  year   = {2026},
  url    = {https://github.com/sanskarpan/probviz}
}
```

Also see [`CITATION.cff`](CITATION.cff).
