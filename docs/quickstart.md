# Quickstart

```bash
git clone https://github.com/sanskarpan/probviz.git
cd probviz
pip install -r requirements.txt
streamlit run web/app.py
```

Open `http://localhost:8501`, pick Continuous/Discrete, choose a distribution, move the
sliders, and inspect the PDF/PMF, CDF, samples, statistics, and quantiles.

## Python API

```python
import numpy as np
from src.distributions import NormalDistribution, BinomialDistribution

normal = NormalDistribution(mu=0, sigma=1)
x = np.linspace(-4, 4, 200)
pdf, cdf = normal.pdf(x), normal.cdf(x)
samples = normal.rvs(size=1000, random_state=42)
print(normal.get_statistics())

binomial = BinomialDistribution(n=10, p=0.3)
print(binomial.pdf(np.arange(0, 11)))
```

## CLI

```bash
pip install -e .
probviz app        # launch the Streamlit UI
probviz test       # run the test suite
probviz version    # print the version
```

## Next steps

- [Installation](installation.md) — Docker, Compose, and dev setup
- [Distributions](distributions.md) — parameters, supports, and recipes
- [Examples](examples.md) — CLT, Binomial→Poisson, comparison plots
