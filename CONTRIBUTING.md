# Contributing

Thanks for considering a contribution. This guide keeps the 695-test suite green
and reviews fast.

## Ground rules

- Be kind and constructive ([Code of Conduct](CODE_OF_CONDUCT.md)).
- Open an issue first for anything beyond a trivial fix so design can be agreed.
- Keep PRs focused; one logical change per PR.
- Add or update tests for every behavior change.
- Update docs (`README.md` / `docs/`) when user-facing behavior changes.

## Setup

```bash
git clone https://github.com/sanskarpan/probviz.git
cd probviz
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Fork and create a branch: `git checkout -b feat/short-name` (or `fix/...`, `docs/...`).
2. Make the change with docstrings + type hints.
3. Add tests under `tests/` (see existing `test_*.py` for patterns).
4. Run the checks below; fix everything before pushing.
5. Push and open a PR against `main` using the PR template.

## Required checks

```bash
pytest tests/ -q
pytest tests/ -q --cov=src --cov-report=term --cov-fail-under=80
flake8 src/ tests/ web/ --count --select=E9,F63,F7,F82 --statistics
mypy --config-file=pyproject.toml src/
python -m black --check --line-length=100 src/ tests/ web/
python -m isort --check --profile=black --line-length=100 src/ tests/ web/
```

Pre-commit runs black/isort/ruff/mypy hooks; CI runs flake8 + mypy + pytest with
coverage + pip-audit + Docker build.

## Style

- Python 3.10+; 100-col lines (black/isort/ruff); type hints on public functions.
- NumPy-style docstrings on public classes/functions.
- Sampling code must use local RNGs (`np.random.default_rng(seed)`) — never mutate
  global `np.random` state.
- Plotting helpers return `(fig, ax)` and never call `plt.show()`.
- Statistical helpers return plain `dict`s.

## Commit messages

Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`,
`ci:`, e.g. `fix: correct EM convergence ordering in MixtureDistribution`.

## Review checklist (for authors)

- [ ] Tests added/updated; `pytest tests/ -q` passes (695+).
- [ ] Coverage gate passes (`--cov-fail-under=80`).
- [ ] Lint + type checks pass.
- [ ] Docs updated (`README.md` / `docs/` / `CHANGELOG.md` under Unreleased if needed).
- [ ] No secrets, no stray files, no `print` debugging.

## Release process (maintainers)

1. Update `CHANGELOG.md` and bump `__version__` (`src/__init__.py`) +
   `pyproject.toml` version together.
2. Tag `vX.Y.Z`; the release workflow builds/pushes Docker, publishes to PyPI,
   deploys docs, and creates the GitHub Release.
