# Statistical Tests (`src.statistical_tests`)

Thin, dict-returning wrappers over SciPy so results are JSON-friendly:

- `hypothesis_tests`: t-test, chi-square, ANOVA, correlation, normality battery.
- `nonparametric`: Mann–Whitney U, Wilcoxon signed-rank, Kruskal–Wallis, Friedman.
- `descriptive`: `describe`, `quantile_summary`, `outlier_detection`,
  `correlation_matrix`.

```python
from probviz.statistical_tests import t_test, describe
print(t_test([1, 2, 3], [1, 2, 4]))
print(describe([1, 2, 3, 100]))
```
