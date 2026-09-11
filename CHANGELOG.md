# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-09-11

### Changed

- Renamed the project and distribution to **`probviz`** (GitHub repo
  `sanskarpan/probviz`, PyPI `probviz`, docs at
  <https://sanskarpan.github.io/probviz/>). All URLs, badges, packaging
  metadata, Docker labels, and citations updated; `docs/archive/` historical
  notes intentionally left untouched.

## [1.0.0] — 2026-09-07

### Added

- `LICENSE` (MIT) and `CITATION.cff`.
- `src/cli.py` with the `probviz` command (`app`, `test`, `version`); fixed the
  packaging entry point (`web.app:main` → `src.cli:main`) and added
  `web/__init__.py` / `src/__init__.py` (`__version__`).
- `tests/conftest.py` centralizing the `src`-layout path setup.
- `requirements-dev.txt` / `requirements-docs.txt` splitting dev and docs tooling.
- `src/visualizers` public facade (`plot_pdf`, `plot_cdf`, `plot_comparison`)
  re-exporting `src.utils.plotting`, resolving the previously empty package.
- Missing `src.utils` re-exports (`validate_integer`,
  `validate_covariance_matrix`, `validate_correlation_matrix`, `log_transform`,
  `box_cox_transform`, `plot_probability_bands`) and abstract-base exports
  (`Copula`, `MultivariateDistribution`).
- MkDocs Material site (`mkdocs.yml`, `docs/`) with GitHub Pages deploy workflow.
- Production repo files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, issue/PR templates, `CODEOWNERS`, `.editorconfig`,
  `.gitattributes`, `MANIFEST.in`, `.github/workflows/docs.yml`,
  `.github/workflows/publish.yml`.

### Fixed

- `MixtureDistribution.fit_em`: convergence is now checked **after** the M-step
  so returned weights/components/responsibilities are never one iteration stale;
  added guards for empty data, `n_components > n`, zero row-likelihoods, zero
  component counts, and zero variances.
- Replaced global `np.random.seed` mutation with local
  `np.random.default_rng(random_state)` in mixture, multivariate-t, and all
  copula samplers (reproducibility preserved per-call).
- Fixed dead `np.atleast_2d` + `ndim == 1` branches in GMM wrappers via a
  `_as_2d` helper that correctly maps 1-D input to `(n, 1)`.
- `plot_bivariate_normal`: builds the 2-D + 3-D panels directly instead of
  orphaning a subplot axis.
- `GaussianCopula.pdf`: clips uniform inputs to avoid `ppf(0/1) → ±inf → nan`.
- `GumbelCopula.rvs`: no longer silently substitutes `0.5` on solver failure;
  degenerate brackets raise `ValueError`, converged roots are kept.
- `fit_copula_to_data`: validates Kendall's τ ranges for Clayton/Gumbel and
  documents the Student-t `df=4` simplification.
- `StudentTCopula`: validates correlation diagonals/positive-definiteness and
  now documents `cdf`/`pdf` as intentionally unimplemented (Monte Carlo via
  `rvs` instead of inheriting a bare `NotImplementedError`).
- `Clayton`/`Gumbel` bivariate-only guards now state the limitation explicitly.
- Removed dead imports (`ProcessPoolExecutor`, unused `warnings`/`optimize`/
  `gammaln`/`Axes3D`) and the redundant inner `Lognormal` numpy import path.

### Changed

- `pyproject.toml` is now the single source of packaging truth (aligned
  runtime deps incl. scikit-learn/statsmodels/joblib, `requires-python >=3.10`,
  ruff config, project URLs); `setup.py` is a thin shim; `requirements.txt`
  mirrors runtime deps.
- Root historical notes moved to `docs/archive/` with a staleness disclaimer.
- `README.md` rewritten for accuracy (16 univariate + advanced modules, real
  tree, correct repo URLs, honest test counts, documented limitations).

### Verification

- `pytest tests/ -q`: **688 passed**.
- `flake8 src/ tests/ --select=E9,F63,F7,F82`, `mypy src/`, Docker build.
