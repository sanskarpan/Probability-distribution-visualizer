# Web App

The Streamlit frontend (`web/app.py`) covers the **16 univariate distributions**:

1. Pick Continuous/Discrete in the sidebar.
2. Choose a distribution; sliders are generated from `get_parameter_bounds()`.
3. Inspect PDF/PMF + CDF (Plotly), random samples vs theory, full statistics,
   and the quantile calculator.

```bash
streamlit run web/app.py
# or
probviz app
```

Advanced modules (multivariate, copulas, mixtures, fitting, Monte Carlo) are
Python-API only by design — the web app stays focused and fast. The plotting
facade used by both the app and library code lives in `src.visualizers`
(`plot_pdf`, `plot_cdf`, `plot_comparison`) over `src.utils.plotting`.
