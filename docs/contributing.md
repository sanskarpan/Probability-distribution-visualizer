# Contributing

Short version of `CONTRIBUTING.md`: fork, branch, add tests, run checks, open a PR.

```bash
pip install -r requirements-dev.txt
pre-commit install
pytest tests/ -q
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --statistics
mypy --config-file=pyproject.toml src/
```

Please read the full [CONTRIBUTING](https://github.com/sanskarpan/probviz/blob/main/CONTRIBUTING.md) guide (style, commit
format, review checklist) before opening a PR.
