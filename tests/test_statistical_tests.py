"""Tests for statistical tests: descriptive, hypothesis, nonparametric."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistical_tests.descriptive import (
    describe,
    quantile_summary,
    outlier_detection,
    correlation_matrix,
)
from statistical_tests.hypothesis_tests import (
    t_test,
    chi_square_test,
    anova,
    correlation_test,
    normality_tests,
)
from statistical_tests.nonparametric import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    friedman_test,
)


# ---------------------------------------------------------------------------
# descriptive.py
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_basic(self):
        np.random.seed(42)
        data = np.random.normal(10, 3, 1000)
        result = describe(data)

        assert 'count' in result
        assert result['count'] == 1000
        assert 'mean' in result
        assert abs(result['mean'] - 10) < 0.3
        assert 'std' in result
        assert abs(result['std'] - 3) < 0.3
        assert 'var' in result
        assert 'se' in result
        assert 'cv' in result
        assert 'min' in result
        assert 'max' in result
        assert 'range' in result
        assert 'median' in result
        assert 'mad' in result
        assert 'q1' in result
        assert 'q3' in result
        assert 'iqr' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'p25' in result
        assert 'p50' in result
        assert 'p75' in result

    def test_describe_custom_percentiles(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = describe(data, percentiles=[10, 90])

        assert 'p10' in result
        assert 'p90' in result
        assert 'p25' not in result

    def test_describe_small_array(self):
        import warnings
        data = np.array([5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = describe(data)

        assert result['count'] == 1
        assert result['mean'] == 5.0

    def test_describe_flat_array(self):
        import warnings
        data = np.array([3.0, 3.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = describe(data)

        assert result['range'] == 0.0

    def test_describe_2d_to_1d(self):
        data = np.array([[1, 2], [3, 4]])
        result = describe(data)

        assert result['count'] == 4

    def test_describe_negative_values(self):
        data = np.array([-5, -2, -1, 0, 1, 2, 5])
        result = describe(data)
        assert result['min'] == -5
        assert result['max'] == 5

    def test_describe_zero_mean_cv(self):
        data = np.array([-1.0, 1.0])
        result = describe(data)
        assert result['cv'] == np.inf


class TestQuantileSummary:
    def test_quantile_summary_quartiles(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = quantile_summary(data, n_quantiles=4)

        assert result['n_quantiles'] == 4
        assert len(result['boundaries']) == 5
        assert len(result['labels']) == len(data)
        assert len(result['counts']) == 4
        assert len(result['proportions']) == 4
        assert np.isclose(np.sum(result['proportions']), 1.0)

    def test_quantile_summary_deciles(self):
        np.random.seed(42)
        data = np.random.uniform(0, 100, 500)
        result = quantile_summary(data, n_quantiles=10)

        assert result['n_quantiles'] == 10
        assert len(result['boundaries']) == 11
        assert len(result['counts']) == 10

    def test_quantile_summary_small_data(self):
        data = np.array([1, 2, 3, 4])
        result = quantile_summary(data, n_quantiles=2)
        assert result['n_quantiles'] == 2

    def test_quantile_summary_uniform_data(self):
        np.random.seed(42)
        data = np.random.uniform(0, 1, 1000)
        result = quantile_summary(data, n_quantiles=4)
        assert all(p > 0.15 for p in result['proportions'])


class TestOutlierDetection:
    def test_outlier_iqr_no_outliers(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = outlier_detection(data, method='iqr', threshold=1.5)

        assert 'method' in result
        assert result['method'] == 'iqr'
        assert result['threshold'] == 1.5
        assert 'n_outliers' in result
        assert 'outlier_indices' in result
        assert 'outlier_values' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert 'is_outlier' in result

    def test_outlier_iqr_with_outliers(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200])
        result = outlier_detection(data, method='iqr', threshold=1.5)

        assert result['n_outliers'] >= 1

    def test_outlier_zscore(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = outlier_detection(data, method='zscore', threshold=3.0)

        assert result['method'] == 'zscore'
        assert 'n_outliers' in result

    def test_outlier_mad(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = outlier_detection(data, method='mad', threshold=3.0)

        assert result['method'] == 'mad'
        assert 'n_outliers' in result

    def test_outlier_mad_constant_data(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = outlier_detection(data, method='mad', threshold=2.5)

        assert result['n_outliers'] == 0

    def test_outlier_unknown_method(self):
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown method"):
            outlier_detection(data, method='invalid')

    def test_outlier_zscore_with_extreme(self):
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 500.0])
        result = outlier_detection(data, method='zscore', threshold=1.5)
        assert result['n_outliers'] >= 1

    def test_outlier_proportion(self):
        data = np.array([1, 2, 3, 100])
        result = outlier_detection(data, method='iqr', threshold=1.5)
        assert 0 <= result['outlier_proportion'] <= 1


class TestCorrelationMatrix:
    def test_correlation_pearson(self):
        np.random.seed(42)
        data = np.column_stack([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
        ])
        result = correlation_matrix(data, method='pearson')

        assert 'correlation_matrix' in result
        assert result['correlation_matrix'].shape == (2, 2)
        assert np.isclose(result['correlation_matrix'][0, 0], 1.0)
        assert np.isclose(result['correlation_matrix'][1, 1], 1.0)

    def test_correlation_spearman(self):
        np.random.seed(42)
        x = np.arange(100)
        y = 2 * x + np.random.normal(0, 5, 100)
        data = np.column_stack([x, y])
        result = correlation_matrix(data, method='spearman')

        assert result['correlation_matrix'].shape == (2, 2)
        assert result['correlation_matrix'][0, 1] > 0.5

    def test_correlation_kendall(self):
        np.random.seed(42)
        x = np.arange(50)
        y = 2 * x + np.random.normal(0, 3, 50)
        data = np.column_stack([x, y])
        result = correlation_matrix(data, method='kendall')

        assert result['correlation_matrix'].shape == (2, 2)

    def test_correlation_with_pvalues(self):
        np.random.seed(42)
        data = np.column_stack([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
        ])
        result = correlation_matrix(data, method='pearson', return_pvalues=True)

        assert 'p_values' in result
        assert result['p_values'].shape == (2, 2)
        assert np.isclose(result['p_values'][0, 0], 0.0)

    def test_correlation_1d_input(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = correlation_matrix(data, method='pearson')

        assert result['correlation_matrix'].shape == (1, 1)
        assert np.isclose(result['correlation_matrix'][0, 0], 1.0)

    def test_correlation_unknown_method(self):
        data = np.column_stack([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="Unknown method"):
            correlation_matrix(data, method='invalid')

    def test_correlation_three_variables(self):
        np.random.seed(42)
        data = np.column_stack([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
        ])
        result = correlation_matrix(data, method='pearson')

        assert result['correlation_matrix'].shape == (3, 3)


# ---------------------------------------------------------------------------
# hypothesis_tests.py
# ---------------------------------------------------------------------------


class TestTTest:
    def test_one_sample_two_sided(self):
        np.random.seed(42)
        sample = np.random.normal(5, 1, 100)
        result = t_test(sample, mu=5, alternative='two-sided')

        assert result['test'] == 'One-sample t-test'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'reject_null' in result
        assert result['alpha'] == 0.05
        assert result['alternative'] == 'two-sided'
        assert result['p_value'] > 0.05

    def test_one_sample_less(self):
        np.random.seed(42)
        sample = np.random.normal(4, 1, 100)
        result = t_test(sample, mu=5, alternative='less')

        assert result['alternative'] == 'less'

    def test_one_sample_greater(self):
        np.random.seed(42)
        sample = np.random.normal(6, 1, 100)
        result = t_test(sample, mu=5, alternative='greater')

        assert result['alternative'] == 'greater'

    def test_two_sample_t_test(self):
        np.random.seed(42)
        group1 = np.random.normal(5, 1, 100)
        group2 = np.random.normal(7, 1, 100)
        result = t_test(group1, group2, alternative='two-sided')

        assert result['test'] == 'Two-sample t-test'
        assert result['p_value'] < 0.05

    def test_two_sample_no_difference(self):
        np.random.seed(42)
        group1 = np.random.normal(5, 1, 100)
        group2 = np.random.normal(5, 1, 100)
        result = t_test(group1, group2, alternative='two-sided')

        assert result['p_value'] > 0.05

    def test_t_test_custom_alpha(self):
        np.random.seed(42)
        sample = np.random.normal(5, 1, 30)
        result = t_test(sample, mu=5, alpha=0.01)
        assert result['alpha'] == 0.01

    def test_t_test_interpretation(self):
        np.random.seed(42)
        sample = np.random.normal(5, 1, 100)
        result = t_test(sample, mu=5)
        assert 'interpretation' in result


class TestChiSquareTest:
    def test_chi_square_uniform_expected(self):
        np.random.seed(42)
        observed = np.array([60, 50, 45, 55, 40])
        result = chi_square_test(observed)

        assert result['test'] == 'Chi-square goodness-of-fit'
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['degrees_of_freedom'] == 4
        assert 'reject_null' in result

    def test_chi_square_custom_expected(self):
        observed = np.array([50, 30, 20])
        expected = np.array([40, 40, 20])
        result = chi_square_test(observed, expected)

        assert result['statistic'] > 0

    def test_chi_square_custom_alpha(self):
        observed = np.array([30, 30, 40])
        result = chi_square_test(observed, alpha=0.01)
        assert result['alpha'] == 0.01

    def test_chi_square_perfect_fit(self):
        observed = np.array([40, 40, 40])
        expected = np.array([40, 40, 40])
        result = chi_square_test(observed, expected)

        assert result['statistic'] == 0.0

    def test_chi_square_interpretation(self):
        observed = np.array([50, 50, 50])
        result = chi_square_test(observed)
        assert 'interpretation' in result


class TestAnova:
    def test_anova_groups_different(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 100)
        g2 = np.random.normal(10, 1, 100)
        g3 = np.random.normal(15, 1, 100)
        result = anova(g1, g2, g3)

        assert result['test'] == 'One-way ANOVA'
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert result['p_value'] < 0.05
        assert result['df_between'] == 2
        assert result['df_within'] == 297
        assert 'conclusion' in result

    def test_anova_groups_same(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 100)
        g2 = np.random.normal(5, 1, 100)
        g3 = np.random.normal(5, 1, 100)
        result = anova(g1, g2, g3)

        assert result['p_value'] > 0.05

    def test_anova_two_groups(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(7, 1, 50)
        result = anova(g1, g2)

        assert result['df_between'] == 1
        assert result['df_within'] == 98

    def test_anova_custom_alpha(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 30)
        g2 = np.random.normal(10, 1, 30)
        result = anova(g1, g2, alpha=0.01)
        assert result['alpha'] == 0.01
        assert result['p_value'] < 0.01

    def test_anova_four_groups(self):
        np.random.seed(42)
        groups = [np.random.normal(m, 1, 50) for m in [5, 8, 11, 14]]
        result = anova(*groups)
        assert result['df_between'] == 3


class TestCorrelationTest:
    def test_correlation_pearson_positive(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.3, 100)
        result = correlation_test(x, y, method='pearson')

        assert result['test'] == 'Pearson correlation'
        assert result['correlation'] > 0.7
        assert result['direction'] == 'positive'
        assert result['strength'] == 'strong'
        assert result['p_value'] < 0.05

    def test_correlation_spearman(self):
        np.random.seed(42)
        x = np.arange(100)
        y = 2 * x + np.random.normal(0, 10, 100)
        result = correlation_test(x, y, method='spearman')

        assert result['test'] == 'Spearman rank correlation'
        assert result['correlation'] > 0.5

    def test_correlation_kendall(self):
        np.random.seed(42)
        x = np.arange(50)
        y = -3 * x + np.random.normal(0, 5, 50)
        result = correlation_test(x, y, method='kendall')

        assert result['test'] == "Kendall's tau"
        assert result['direction'] == 'negative'

    def test_correlation_no_correlation(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        result = correlation_test(x, y, method='pearson')

        assert result['p_value'] > 0.05

    def test_correlation_weak_strength(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = 0.1 * x + np.random.normal(0, 1, 200)
        result = correlation_test(x, y, method='pearson')

        assert result['strength'] == 'weak'

    def test_correlation_moderate_strength(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = 0.5 * x + np.random.normal(0, 1, 200)
        result = correlation_test(x, y, method='pearson')

        assert result['strength'] == 'moderate'

    def test_correlation_unknown_method(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        with pytest.raises(ValueError, match="Unknown method"):
            correlation_test(x, y, method='invalid')

    def test_correlation_custom_alpha(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.5, 100)
        result = correlation_test(x, y, method='pearson', alpha=0.01)
        assert result['alpha'] == 0.01


class TestNormalityTests:
    def test_normality_normal_data(self):
        import warnings
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = normality_tests(data)

        assert 'shapiro_wilk' in result
        assert 'kolmogorov_smirnov' in result
        assert 'anderson_darling' in result
        assert 'jarque_bera' in result
        assert 'consensus' in result
        assert result['consensus']['conclusion'] == 'Likely normal'

    def test_normality_exponential_data(self):
        import warnings
        np.random.seed(42)
        data = np.random.exponential(1, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = normality_tests(data)

        assert result['shapiro_wilk']['p_value'] < 0.05
        assert result['consensus']['conclusion'] == 'Likely non-normal'

    def test_normality_large_sample_skips_sw(self):
        import warnings
        np.random.seed(42)
        data = np.random.normal(0, 1, 6000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = normality_tests(data)

        assert 'shapiro_wilk' not in result
        assert 'kolmogorov_smirnov' in result
        assert 'jarque_bera' in result

    def test_normality_custom_alpha(self):
        import warnings
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = normality_tests(data, alpha=0.01)
        assert result['jarque_bera']['p_value'] > 0.01

    def test_normality_anderson_critical(self):
        import warnings
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = normality_tests(data)
        assert 'critical_value' in result['anderson_darling']
        assert 'significance_level' in result['anderson_darling']

    def test_normality_consensus_mixed(self):
        import warnings
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = normality_tests(data, alpha=0.05)

        assert 'tests_rejecting_normality' in result['consensus']
        assert 'total_tests' in result['consensus']


# ---------------------------------------------------------------------------
# nonparametric.py
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    def test_mann_whitney_different_distributions(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 100)
        g2 = np.random.normal(7, 1, 100)
        result = mann_whitney_u(g1, g2)

        assert result['test'] == 'Mann-Whitney U test'
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['p_value'] < 0.05
        assert result['reject_null']
        assert result['alternative'] == 'two-sided'
        assert 'conclusion' in result

    def test_mann_whitney_same_distributions(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 100)
        g2 = np.random.normal(5, 1, 100)
        result = mann_whitney_u(g1, g2)

        assert result['p_value'] > 0.05
        assert not result['reject_null']

    def test_mann_whitney_alternative_less(self):
        np.random.seed(42)
        g1 = np.random.normal(3, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        result = mann_whitney_u(g1, g2, alternative='less')

        assert result['alternative'] == 'less'

    def test_mann_whitney_alternative_greater(self):
        np.random.seed(42)
        g1 = np.random.normal(7, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        result = mann_whitney_u(g1, g2, alternative='greater')

        assert result['alternative'] == 'greater'

    def test_mann_whitney_custom_alpha(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(7, 1, 50)
        result = mann_whitney_u(g1, g2, alpha=0.01)
        assert result['alpha'] == 0.01

    def test_mann_whitney_interpretation(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 30)
        g2 = np.random.normal(5, 1, 30)
        result = mann_whitney_u(g1, g2)
        assert 'interpretation' in result

    def test_mann_whitney_small_samples(self):
        np.random.seed(42)
        g1 = np.array([1, 2, 3, 4, 5])
        g2 = np.array([10, 11, 12, 13, 14])
        result = mann_whitney_u(g1, g2)
        assert result['p_value'] < 0.05


class TestWilcoxonSignedRank:
    def test_wilcoxon_two_samples(self):
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        after = baseline + np.random.normal(2, 0.5, 50)
        result = wilcoxon_signed_rank(after, baseline)

        assert result['test'] == 'Wilcoxon signed-rank test'
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['p_value'] < 0.05
        assert 'n_differences' in result

    def test_wilcoxon_differences_as_input(self):
        np.random.seed(42)
        diffs = np.random.normal(2, 1, 50)
        result = wilcoxon_signed_rank(diffs, sample2=None)

        assert result['test'] == 'Wilcoxon signed-rank test'
        assert result['p_value'] < 0.05

    def test_wilcoxon_no_difference(self):
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        after = baseline + np.random.normal(0, 0.1, 50)
        result = wilcoxon_signed_rank(after, baseline)

        assert result['p_value'] > 0.05

    def test_wilcoxon_all_zero_differences(self):
        g1 = np.array([5, 5, 5, 5])
        g2 = np.array([5, 5, 5, 5])
        result = wilcoxon_signed_rank(g1, g2)

        assert 'error' in result
        assert result['error'] == 'No non-zero differences found'
        assert np.isnan(result['statistic'])
        assert np.isnan(result['p_value'])

    def test_wilcoxon_alternative_less(self):
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        after = baseline - np.random.normal(2, 0.5, 50)
        result = wilcoxon_signed_rank(baseline, after, alternative='less')

        assert result['alternative'] == 'less'

    def test_wilcoxon_alternative_greater(self):
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        after = baseline + np.random.normal(2, 0.5, 50)
        result = wilcoxon_signed_rank(after, baseline, alternative='greater')

        assert result['alternative'] == 'greater'

    def test_wilcoxon_custom_alpha(self):
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        after = baseline + np.random.normal(2, 0.5, 50)
        result = wilcoxon_signed_rank(after, baseline, alpha=0.01)
        assert result['alpha'] == 0.01

    def test_wilcoxon_some_zero_differences(self):
        np.random.seed(42)
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
        g2 = np.array([1.5, 2.5, 3.0, 4.5, 5.5, 5.0])
        result = wilcoxon_signed_rank(g1, g2)
        assert 'error' not in result


class TestKruskalWallis:
    def test_kruskal_wallis_groups_different(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(10, 1, 50)
        g3 = np.random.normal(15, 1, 50)
        result = kruskal_wallis(g1, g2, g3)

        assert result['test'] == 'Kruskal-Wallis H test'
        assert 'h_statistic' in result
        assert 'p_value' in result
        assert result['p_value'] < 0.05
        assert result['degrees_of_freedom'] == 2
        assert result['n_groups'] == 3

    def test_kruskal_wallis_groups_same(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        g3 = np.random.normal(5, 1, 50)
        result = kruskal_wallis(g1, g2, g3)

        assert result['p_value'] > 0.05

    def test_kruskal_wallis_two_groups(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 100)
        g2 = np.random.normal(8, 1, 100)
        result = kruskal_wallis(g1, g2)

        assert result['degrees_of_freedom'] == 1
        assert result['n_groups'] == 2
        assert result['p_value'] < 0.05

    def test_kruskal_wallis_custom_alpha(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(10, 1, 50)
        result = kruskal_wallis(g1, g2, alpha=0.01)
        assert result['alpha'] == 0.01

    def test_kruskal_wallis_non_normal_data(self):
        np.random.seed(42)
        g1 = np.random.exponential(1, 50)
        g2 = np.random.exponential(5, 50)
        g3 = np.random.exponential(10, 50)
        result = kruskal_wallis(g1, g2, g3)
        assert result['p_value'] < 0.05

    def test_kruskal_wallis_interpretation(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 30)
        g2 = np.random.normal(10, 1, 30)
        result = kruskal_wallis(g1, g2)
        assert 'interpretation' in result
        assert 'conclusion' in result


class TestFriedmanTest:
    def test_friedman_groups_different(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(10, 1, 50)
        g3 = np.random.normal(15, 1, 50)
        result = friedman_test(g1, g2, g3)

        assert result['test'] == 'Friedman test'
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['p_value'] < 0.05
        assert result['degrees_of_freedom'] == 2
        assert result['n_treatments'] == 3
        assert result['n_blocks'] == 50

    def test_friedman_groups_same(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        g3 = np.random.normal(5, 1, 50)
        result = friedman_test(g1, g2, g3)

        assert result['p_value'] > 0.05

    def test_friedman_two_groups(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(8, 1, 50)
        with pytest.raises(ValueError, match="At least 3 samples"):
            friedman_test(g1, g2)

    def test_friedman_unequal_lengths_error(self):
        g1 = np.array([1, 2, 3, 4, 5])
        g2 = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="same length"):
            friedman_test(g1, g2)

    def test_friedman_interpretation(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 30)
        g2 = np.random.normal(10, 1, 30)
        g3 = np.random.normal(15, 1, 30)
        result = friedman_test(g1, g2, g3)
        assert 'interpretation' in result
        assert 'conclusion' in result

    def test_friedman_many_treatments(self):
        np.random.seed(42)
        groups = [np.random.normal(m, 1, 30) for m in [3, 5, 7, 9, 11]]
        result = friedman_test(*groups)
        assert result['n_treatments'] == 5
        assert result['p_value'] < 0.05
