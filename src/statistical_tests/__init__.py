"""Statistical tests and hypothesis testing utilities.

Note: This module is named 'statistical_tests' to avoid collision
with Python's built-in 'statistics' module.
"""

from .hypothesis_tests import (
    t_test,
    chi_square_test,
    anova,
    correlation_test,
    normality_tests,
)
from .descriptive import (
    describe,
    quantile_summary,
    outlier_detection,
    correlation_matrix,
)
from .nonparametric import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    friedman_test,
)

__all__ = [
    # Hypothesis tests
    "t_test",
    "chi_square_test",
    "anova",
    "correlation_test",
    "normality_tests",
    # Descriptive statistics
    "describe",
    "quantile_summary",
    "outlier_detection",
    "correlation_matrix",
    # Nonparametric tests
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    "friedman_test",
]
