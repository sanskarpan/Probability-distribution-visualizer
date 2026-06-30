"""Nonparametric statistical tests."""

from typing import Dict, Union, Optional
import numpy as np
from scipy import stats


def mann_whitney_u(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Mann-Whitney U test (nonparametric alternative to two-sample t-test).

    Args:
        sample1: First sample
        sample2: Second sample
        alternative: 'two-sided', 'less', or 'greater'
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.mannwhitneyu(
        sample1, sample2,
        alternative=alternative
    )

    reject_null = p_value < alpha

    return {
        'test': 'Mann-Whitney U test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': alternative,
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis",
        'conclusion': f"Distributions are {'different' if reject_null else 'not significantly different'}"
    }


def wilcoxon_signed_rank(
    sample1: np.ndarray,
    sample2: Optional[np.ndarray] = None,
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Wilcoxon signed-rank test (nonparametric paired test).

    Args:
        sample1: First sample or differences
        sample2: Second sample (None if sample1 contains differences)
        alternative: 'two-sided', 'less', or 'greater'
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if sample2 is not None:
        # Compute differences
        differences = sample1 - sample2
    else:
        differences = sample1

    # Remove zeros
    differences = differences[differences != 0]

    if len(differences) == 0:
        return {
            'test': 'Wilcoxon signed-rank test',
            'error': 'No non-zero differences found',
            'statistic': np.nan,
            'p_value': np.nan,
            'reject_null': False
        }

    statistic, p_value = stats.wilcoxon(
        differences,
        alternative=alternative
    )

    reject_null = p_value < alpha

    return {
        'test': 'Wilcoxon signed-rank test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': alternative,
        'n_differences': len(differences),
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis",
        'conclusion': f"Paired observations are {'different' if reject_null else 'not significantly different'}"
    }


def kruskal_wallis(
    *samples: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, int]]:
    """
    Kruskal-Wallis H test (nonparametric alternative to one-way ANOVA).

    Args:
        *samples: Variable number of sample groups
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.kruskal(*samples)
    reject_null = p_value < alpha

    df = len(samples) - 1

    return {
        'test': 'Kruskal-Wallis H test',
        'h_statistic': float(statistic),
        'p_value': float(p_value),
        'degrees_of_freedom': df,
        'n_groups': len(samples),
        'reject_null': reject_null,
        'alpha': alpha,
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis",
        'conclusion': f"Group distributions are {'different' if reject_null else 'not significantly different'}"
    }


def friedman_test(
    *samples: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, int]]:
    """
    Friedman test (nonparametric alternative to repeated measures ANOVA).

    Args:
        *samples: Variable number of sample groups (must have same length)
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Check that all samples have same length
    lengths = [len(s) for s in samples]
    if len(set(lengths)) > 1:
        raise ValueError("All samples must have the same length for Friedman test")

    statistic, p_value = stats.friedmanchisquare(*samples)
    reject_null = p_value < alpha

    k = len(samples)  # number of treatments
    n = lengths[0]    # number of blocks
    df = k - 1

    return {
        'test': 'Friedman test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'degrees_of_freedom': df,
        'n_treatments': k,
        'n_blocks': n,
        'reject_null': reject_null,
        'alpha': alpha,
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis",
        'conclusion': f"Treatment effects are {'different' if reject_null else 'not significantly different'}"
    }


# Import for type hints is already at top of file
