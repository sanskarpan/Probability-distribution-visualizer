# Quick Start Guide

## Fastest Way to Get Started

### Option 1: Using the Run Script (Recommended)

```bash
cd Probability-distribution-visualizer
./run.sh
```

Then select:
- Option 1 for Web Interface
- Option 2 for Examples
- Option 3 for Tests

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch web interface
streamlit run web/app.py
```

## What Can You Do?

### Web Interface Features

1. **Select Distribution Type**: Choose between Continuous or Discrete
2. **Pick a Distribution**: 16 distributions available
3. **Adjust Parameters**: Use interactive sliders
4. **Visualize**: See PDF/PMF and CDF in real-time
5. **Generate Samples**: Create random samples and compare with theory
6. **View Statistics**: Mean, variance, skewness, kurtosis, and more
7. **Calculate Quantiles**: Find percentiles instantly

### Available Distributions

**Continuous (10):**
- Normal, Exponential, Uniform, Beta, Gamma
- Chi-Square, Student-t, Weibull, Lognormal, Cauchy

**Discrete (6):**
- Binomial, Poisson, Geometric
- Negative Binomial, Hypergeometric, Discrete Uniform

## Example Code

```python
from distributions import NormalDistribution
import numpy as np

# Create distribution
normal = NormalDistribution(mu=0, sigma=1)

# Get statistics
print(f"Mean: {normal.mean()}")
print(f"Std Dev: {normal.std()}")

# Calculate PDF
x = np.linspace(-3, 3, 100)
pdf = normal.pdf(x)

# Generate samples
samples = normal.rvs(size=1000)
```

## Troubleshooting

### Dependencies Not Installing?

Try:
```bash
pip install --upgrade pip
pip install --user -r requirements.txt
```

### Streamlit Not Found?

```bash
pip install streamlit
```

### Import Errors?

Make sure you're in the virtual environment:
```bash
source venv/bin/activate
```

## Next Steps

1. Explore the web interface at `http://localhost:8501`
2. Try the examples: `python examples/basic_usage.py`
3. Read the full README.md for detailed documentation
4. Check out the API in `src/distributions/`

## Tips

- **Best Experience**: Use a modern browser (Chrome, Firefox, Safari)
- **Performance**: The web app is optimized for smooth interaction
- **Learning**: Try comparing different parameter values to understand distributions
- **Export**: Use your browser's screenshot tool to save visualizations

Enjoy exploring probability distributions!
