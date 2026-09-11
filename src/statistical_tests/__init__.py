"""Statistical tests and hypothesis testing utilities.

Note: This module is named 'statistical_tests' to avoid collision
with Python's built-in 'statistics' module.
"""

from .descriptive import (
    correlation_matrix,
    describe,
    outlier_detection,
    quantile_summary,
)
from .hypothesis_tests import (
    anova,
    chi_square_test,
    correlation_test,
    normality_tests,
    t_test,
)
from .nonparametric import (
    friedman_test,
    kruskal_wallis,
    mann_whitney_u,
    wilcoxon_signed_rank,
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
