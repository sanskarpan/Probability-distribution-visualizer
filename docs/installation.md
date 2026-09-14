# Installation

## Requirements

- Python 3.10, 3.11, or 3.12 (CI-tested)
- pip 23+

## From PyPI

```bash
pip install probviz
probviz app
```

## From source (recommended for the web app)

```bash
git clone https://github.com/sanskarpan/probviz.git
cd probviz
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run web/app.py
```

## Editable install (Python API + CLI)

```bash
pip install -e ".[dev]"
probviz app
```

## Docker

```bash
docker build -t probviz .
docker run -p 8501:8501 probviz
# or
docker compose up --build
```

The image runs as non-root `appuser`, exposes `8501`, and includes a
`/_stcore/health` healthcheck.

## Dependency groups

| File | Purpose |
|---|---|
| `requirements.txt` | runtime (numpy, scipy, pandas, matplotlib, seaborn, plotly, streamlit, scikit-learn, statsmodels, joblib) |
| `requirements-dev.txt` | lint/type/test tooling |
| `requirements-docs.txt` | MkDocs site |

Canonical metadata lives in `pyproject.toml`; `requirements*.txt` mirror it for
pip/Docker workflows. `setup.py` is a thin shim only.
