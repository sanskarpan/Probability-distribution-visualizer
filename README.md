# Probability Distribution Visualizer

An interactive, comprehensive probability distribution visualizer with a stunning web interface built with Streamlit. Explore, visualize, and understand probability distributions through interactive charts and real-time parameter adjustments.

<p align="center">
  <a href="https://github.com/sanskarpan/Probability-distribution-visualizer/actions/workflows/ci.yml"><img src="https://github.com/sanskarpan/Probability-distribution-visualizer/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/sanskarpan/Probability-distribution-visualizer/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-688%20passed-brightgreen" alt="Tests"></a>
  <a href="https://github.com/sanskarpan/Probability-distribution-visualizer/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-96%25-brightgreen" alt="Coverage"></a>
  <a href="https://github.com/sanskarpan/Probability-distribution-visualizer/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="https://github.com/sanskarpan/Probability-distribution-visualizer"><img src="https://img.shields.io/github/stars/sanskarpan/Probability-distribution-visualizer?style=social" alt="Stars"></a>
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?logo=streamlit" alt="Streamlit"></a>
  <a href="https://docker.com"><img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker" alt="Docker"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://github.com/sanskarpan/Probability-distribution-visualizer/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome"></a>
</p>

## Features

### 🎯 16 Probability Distributions

**Continuous Distributions:**
- Normal (Gaussian)
- Exponential
- Uniform
- Beta
- Gamma
- Chi-Square
- Student-t
- Weibull
- Lognormal
- Cauchy

**Discrete Distributions:**
- Binomial
- Poisson
- Geometric
- Negative Binomial
- Hypergeometric
- Discrete Uniform

### ⚡ Key Features

- **Interactive Web Interface**: Beautiful, modern Streamlit interface with gradient designs
- **Real-time Visualization**: Instantly see how parameter changes affect distributions
- **Comprehensive Statistics**: Mean, variance, std dev, median, mode, skewness, kurtosis
- **PDF/PMF & CDF Plots**: Interactive Plotly charts with hover information
- **Random Sampling**: Generate and visualize random samples with theoretical overlay
- **Quantile Calculator**: Calculate percentiles and quantiles
- **Parameter Validation**: Smart bounds and validation for all parameters
- **Export Ready**: Publication-quality plots ready for export

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/probability-distribution-visualizer.git
cd probability-distribution-visualizer

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run web/app.py
```

### Install as Package

```bash
pip install -e .
```

## Usage

### Web Application (Recommended)

Launch the interactive web interface:

```bash
streamlit run web/app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
1. Select distribution type (Continuous/Discrete)
2. Choose from 16 available distributions
3. Adjust parameters with interactive sliders
4. View PDF/PMF and CDF plots in real-time
5. Generate random samples and compare with theoretical distribution
6. Explore comprehensive statistics
7. Calculate quantiles and percentiles

### Python API

```python
from distributions import NormalDistribution, BinomialDistribution
import numpy as np

# Create a normal distribution
normal = NormalDistribution(mu=0, sigma=1)

# Get statistics
print(f"Mean: {normal.mean()}")
print(f"Variance: {normal.var()}")
print(f"Std Dev: {normal.std()}")

# Calculate PDF
x = np.linspace(-4, 4, 100)
pdf_values = normal.pdf(x)

# Calculate CDF
cdf_values = normal.cdf(x)

# Generate random samples
samples = normal.rvs(size=1000, random_state=42)

# Get quantiles
median = normal.ppf(0.5)
q95 = normal.ppf(0.95)

# Get comprehensive statistics
stats = normal.get_statistics()
print(stats)

# Discrete example: Binomial distribution
binomial = BinomialDistribution(n=10, p=0.3)
pmf_values = binomial.pdf(np.arange(0, 11))
```

## Distribution Details

### Continuous Distributions

#### Normal Distribution
- **Parameters**: μ (mean), σ (standard deviation)
- **Use Case**: Natural phenomena, measurement errors, Central Limit Theorem
- **Support**: (-∞, ∞)

#### Exponential Distribution
- **Parameters**: λ (rate)
- **Use Case**: Time between events, survival analysis, reliability engineering
- **Support**: [0, ∞)

#### Beta Distribution
- **Parameters**: α (alpha), β (beta)
- **Use Case**: Modeling proportions, Bayesian statistics, success rates
- **Support**: [0, 1]

#### Gamma Distribution
- **Parameters**: shape (k), scale (θ)
- **Use Case**: Waiting times, insurance claims, rainfall analysis
- **Support**: [0, ∞)

### Discrete Distributions

#### Binomial Distribution
- **Parameters**: n (trials), p (success probability)
- **Use Case**: Number of successes in fixed trials, A/B testing
- **Support**: {0, 1, ..., n}

#### Poisson Distribution
- **Parameters**: λ (rate/mean)
- **Use Case**: Count data, rare events, arrivals per time period
- **Support**: {0, 1, 2, ...}

#### Geometric Distribution
- **Parameters**: p (success probability)
- **Use Case**: Number of trials until first success
- **Support**: {1, 2, 3, ...}

## Project Structure

```
Probability-distribution-visualizer/
├── src/
│   ├── distributions/
│   │   ├── __init__.py
│   │   ├── base.py           # Base Distribution class
│   │   ├── continuous.py     # 10 continuous distributions
│   │   └── discrete.py       # 6 discrete distributions
│   ├── visualizers/
│   │   ├── __init__.py
│   │   ├── pdf_visualizer.py
│   │   ├── cdf_visualizer.py
│   │   └── comparison.py
│   ├── statistics/
│   │   ├── __init__.py
│   │   └── calculator.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── web/
│   └── app.py               # Streamlit web application
├── tests/
│   ├── test_distributions.py
│   ├── test_visualizers.py
│   └── test_statistics.py
├── examples/
│   ├── basic_usage.py
│   ├── comparison.py
│   └── simulation.py
├── docs/
│   └── images/
├── requirements.txt
├── setup.py
└── README.md
```

## Examples

### Example 1: Compare Normal Distributions

```python
from distributions import NormalDistribution
import matplotlib.pyplot as plt
import numpy as np

# Create distributions with different parameters
norm1 = NormalDistribution(mu=0, sigma=1)
norm2 = NormalDistribution(mu=2, sigma=1.5)
norm3 = NormalDistribution(mu=-1, sigma=0.5)

# Generate x values
x = np.linspace(-5, 6, 500)

# Plot PDFs
plt.figure(figsize=(12, 6))
plt.plot(x, norm1.pdf(x), label=f'N(0, 1)', linewidth=2)
plt.plot(x, norm2.pdf(x), label=f'N(2, 1.5)', linewidth=2)
plt.plot(x, norm3.pdf(x), label=f'N(-1, 0.5)', linewidth=2)

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Comparison of Normal Distributions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: Central Limit Theorem Demonstration

```python
from distributions import UniformDistribution, NormalDistribution
import numpy as np
import matplotlib.pyplot as plt

# Start with uniform distribution
uniform = UniformDistribution(a=0, b=1)

# Sample means
sample_sizes = [1, 5, 10, 30]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, n in enumerate(sample_sizes):
    ax = axes[idx // 2, idx % 2]

    # Generate sample means
    sample_means = []
    for _ in range(10000):
        samples = uniform.rvs(size=n)
        sample_means.append(np.mean(samples))

    # Plot histogram
    ax.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Overlay normal distribution
    x = np.linspace(min(sample_means), max(sample_means), 100)
    normal = NormalDistribution(
        mu=np.mean(sample_means),
        sigma=np.std(sample_means)
    )
    ax.plot(x, normal.pdf(x), 'r-', linewidth=2, label='Normal fit')

    ax.set_title(f'Sample Size n={n}')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Central Limit Theorem Demonstration', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Example 3: Binomial vs Poisson Approximation

```python
from distributions import BinomialDistribution, PoissonDistribution
import numpy as np
import matplotlib.pyplot as plt

# Binomial with large n and small p
n, p = 100, 0.03
binomial = BinomialDistribution(n=n, p=p)

# Poisson approximation (lambda = n*p)
poisson = PoissonDistribution(lambda_param=n * p)

# Plot
x = np.arange(0, 15)
plt.figure(figsize=(12, 6))

plt.bar(x - 0.2, binomial.pdf(x), width=0.4, label='Binomial(100, 0.03)', alpha=0.7)
plt.bar(x + 0.2, poisson.pdf(x), width=0.4, label='Poisson(3)', alpha=0.7)

plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Poisson Approximation to Binomial')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## API Documentation

### Distribution Base Class

All distributions inherit from the `Distribution` base class:

```python
class Distribution(ABC):
    def pdf(x):              # Probability density/mass function
    def cdf(x):              # Cumulative distribution function
    def ppf(q):              # Percent point function (inverse CDF)
    def rvs(size, random_state):  # Random samples
    def mean():              # Mean
    def var():               # Variance
    def std():               # Standard deviation
    def median():            # Median
    def mode():              # Mode
    def skewness():          # Skewness
    def kurtosis():          # Excess kurtosis
    def entropy():           # Differential entropy
    def interval(alpha):     # Confidence interval
    def get_statistics():    # All statistics as dict
```

## Performance

- **Fast computation**: Leverages NumPy and SciPy optimized routines
- **Responsive UI**: Streamlit with Plotly for smooth interactivity
- **Memory efficient**: Lazy evaluation and smart caching
- **Scalable**: Handle thousands of data points smoothly

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass
6. Submit a pull request

## Educational Use

This tool is perfect for:

- **Statistics courses**: Interactive exploration of distributions
- **Data science education**: Understanding probabilistic models
- **Research**: Quick visualization and parameter exploration
- **Self-learning**: Hands-on probability theory practice

## Future Enhancements

- [ ] Add multivariate distributions (Multivariate Normal, Dirichlet)
- [ ] Distribution fitting from data
- [ ] Hypothesis testing tools
- [ ] Confidence interval visualization
- [ ] Distribution comparison metrics (KL divergence, Wasserstein distance)
- [ ] Export to various formats (PNG, PDF, LaTeX)
- [ ] Animation of parameter changes
- [ ] Mobile-responsive design

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [SciPy](https://scipy.org/) for statistical computations
- Visualizations powered by [Plotly](https://plotly.com/)
- Web interface created with [Streamlit](https://streamlit.io/)
- Inspired by statistical education and interactive learning

## Citation

If you use this tool in your research or teaching, please cite:

```bibtex
@software{probability_distribution_visualizer,
  title = {Probability Distribution Visualizer},
  author = {Research Project},
  year = {2024},
  url = {https://github.com/yourusername/probability-distribution-visualizer}
}
```

---

**Built with ❤️ for statistics education and data science**

