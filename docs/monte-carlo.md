# Monte Carlo (`src.monte_carlo`)

- `MonteCarloSimulator(random_seed)` — `simulate`, `estimate_probability`,
  `estimate_expectation`, `importance_sampling`, `stratified_sampling`,
  `bootstrap`, `permutation_test`, with convergence tracking in
  `SimulationResult`.
- `VarianceReduction` — antithetic variates, control variates.
- `QuasiMonteCarloSimulator` — Halton/Sobol sequences and QMC integration.

```python
from src.monte_carlo import MonteCarloSimulator
import numpy as np

rng = np.random.default_rng(42)
sim = MonteCarloSimulator(random_seed=42)
res = sim.estimate_probability(lambda: rng.normal(0, 1) > 1.0, num_samples=100_000)
print(res["probability"], res["confidence_interval"])
```
