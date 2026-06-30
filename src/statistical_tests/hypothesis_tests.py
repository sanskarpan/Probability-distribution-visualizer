"""Hypothesis testing functions."""

from typing import Dict, Optional, Union, Tuple
import numpy as np
from scipy import stats


def t_test(
    sample1: np.ndarray,
    sample2: Optional[np.ndarray] = None,
    mu: float = 0,
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Perform t-test (one-sample or two-sample).

    Args:
        sample1: First sample data
        sample2: Second sample data (None for one-sample test)
        mu: Hypothesized mean (for one-sample test)
        alternative: 'two-sided', 'less', or 'greater'
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if sample2 is None:
        # One-sample t-test
        statistic, p_value = stats.ttest_1samp(sample1, mu, alternative=alternative)
        test_type = "One-sample t-test"
    else:
        # Two-sample t-test
        statistic, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
        test_type = "Two-sample t-test"

    reject_null = p_value < alpha

    return {
        'test': test_type,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': alternative,
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis at α={alpha}"
    }


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, int]]:
    """
    Perform chi-square goodness-of-fit test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies (uniform if None)
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if expected is None:
        # Assume uniform distribution
        expected = np.ones_like(observed) * np.mean(observed)

    statistic, p_value = stats.chisquare(observed, expected)
    df = len(observed) - 1
    reject_null = p_value < alpha

    return {
        'test': 'Chi-square goodness-of-fit',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'degrees_of_freedom': df,
        'reject_null': reject_null,
        'alpha': alpha,
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis at α={alpha}"
    }


def anova(
    *samples: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """
    Perform one-way ANOVA.

    Args:
        *samples: Variable number of sample groups
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.f_oneway(*samples)
    reject_null = p_value < alpha

    df_between = len(samples) - 1
    df_within = sum(len(s) for s in samples) - len(samples)

    return {
        'test': 'One-way ANOVA',
        'f_statistic': float(statistic),
        'p_value': float(p_value),
        'df_between': df_between,
        'df_within': df_within,
        'reject_null': reject_null,
        'alpha': alpha,
        'interpretation': f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis at α={alpha}",
        'conclusion': f"Group means are {'significantly different' if reject_null else 'not significantly different'}"
    }


def correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Test for correlation between two variables.

    Args:
        x: First variable
        y: Second variable
        method: 'pearson', 'spearman', or 'kendall'
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if method == 'pearson':
        statistic, p_value = stats.pearsonr(x, y)
        test_name = "Pearson correlation"
    elif method == 'spearman':
        statistic, p_value = stats.spearmanr(x, y)
        test_name = "Spearman rank correlation"
    elif method == 'kendall':
        statistic, p_value = stats.kendalltau(x, y)
        test_name = "Kendall's tau"
    else:
        raise ValueError(f"Unknown method: {method}")

    reject_null = p_value < alpha

    # Interpret strength
    abs_corr = abs(statistic)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if statistic > 0 else "negative"

    return {
        'test': test_name,
        'correlation': float(statistic),
        'p_value': float(p_value),
        'reject_null': reject_null,
        'alpha': alpha,
        'interpretation': f"{'Significant' if reject_null else 'Non-significant'} {strength} {direction} correlation",
        'strength': strength,
        'direction': direction
    }


def normality_tests(
    data: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Dict[str, Union[float, bool]]]:
    """
    Run multiple normality tests.

    Args:
        data: Sample data
        alpha: Significance level

    Returns:
        Dictionary with results from multiple tests
    """
    results = {}

    # Shapiro-Wilk test
    if len(data) <= 5000:  # SW test has sample size limits
        sw_stat, sw_p = stats.shapiro(data)
        results['shapiro_wilk'] = {
            'statistic': float(sw_stat),
            'p_value': float(sw_p),
            'reject_null': sw_p < alpha,
            'conclusion': 'Non-normal' if sw_p < alpha else 'Normal'
        }

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    results['kolmogorov_smirnov'] = {
        'statistic': float(ks_stat),
        'p_value': float(ks_p),
        'reject_null': ks_p < alpha,
        'conclusion': 'Non-normal' if ks_p < alpha else 'Normal'
    }

    # Anderson-Darling test
    ad_result = stats.anderson(data, dist='norm')
    # Find critical value for given alpha
    critical_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}.get(alpha, 2)
    results['anderson_darling'] = {
        'statistic': float(ad_result.statistic),
        'critical_value': float(ad_result.critical_values[critical_idx]),
        'significance_level': float(ad_result.significance_level[critical_idx]),
        'reject_null': ad_result.statistic > ad_result.critical_values[critical_idx],
        'conclusion': 'Non-normal' if ad_result.statistic > ad_result.critical_values[critical_idx] else 'Normal'
    }

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(data)
    results['jarque_bera'] = {
        'statistic': float(jb_stat),
        'p_value': float(jb_p),
        'reject_null': jb_p < alpha,
        'conclusion': 'Non-normal' if jb_p < alpha else 'Normal'
    }

    # Overall consensus
    reject_count = sum(1 for r in results.values() if r.get('reject_null', False))
    total_tests = len(results)

    results['consensus'] = {
        'tests_rejecting_normality': reject_count,
        'total_tests': total_tests,
        'conclusion': 'Likely non-normal' if reject_count > total_tests / 2 else 'Likely normal'
    }

    return results
