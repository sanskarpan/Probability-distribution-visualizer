# Quickstart

Get from zero to interactive plots in under five minutes.

## 1. Install

```bash
git clone https://github.com/sanskarpan/probviz.git
cd probviz
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Launch the web app

```bash
streamlit run web/app.py
# or, after `pip install -e .`:
probviz app
```

Open `http://localhost:8501`.

## 3. Explore

1. Pick **Continuous** or **Discrete** in the sidebar.
2. Choose one of the **16 univariate distributions**.
3. Move the parameter sliders (bounds come from `get_parameter_bounds()`).
4. Read the PDF/PMF + CDF charts, sample overlay, statistics table, and quantiles.

## 4. Try the Python API

```python
import numpy as np
from src.distributions import NormalDistribution

normal = NormalDistribution(mu=0, sigma=1)
x = np.linspace(-4, 4, 200)
pdf, cdf = normal.pdf(x), normal.cdf(x)
samples = normal.rvs(size=1000, random_state=42)
print(normal.get_statistics())
```

## 5. Go further

- `docs/` (GitHub Pages): fitting, Monte Carlo, copulas, mixtures, API reference.
- `examples/basic_usage.py`: `python examples/basic_usage.py`.
- `pytest tests/ -q`: run the 688-test suite.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit: command not found` | Activate the venv / `pip install -r requirements.txt` |
| Port 8501 busy | `streamlit run web/app.py --server.port 8502` |
| Import errors in snippets | Run from the repo root or `pip install -e .`; imports are `from src....` |
