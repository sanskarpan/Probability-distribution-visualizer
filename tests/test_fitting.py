"""Tests for distribution fitting."""

import warnings
import warnings
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fitting.distribution_fitter import (
    DistributionFitter,
    BayesianEstimator,
    GoodnessOfFit,
)


class TestDistributionFitter:
    """Test distribution fitting functionality."""

    def test_creation(self):
        """Test fitter creation."""
        data = np.random.normal(0, 1, 100)
        fitter = DistributionFitter(data)
        assert len(fitter.data) == 100

    def test_fit_distribution(self):
        """Test fitting a single distribution."""
        np.random.seed(42)
        data = np.random.normal(5, 2, 1000)

        fitter = DistributionFitter(data)
        result = fitter.fit_distribution('norm')

        assert 'parameters' in result
        assert 'aic' in result
        assert 'bic' in result
        assert 'log_likelihood' in result

        loc, scale = result['parameters']
        assert abs(loc - 5) < 0.5
        assert abs(scale - 2) < 0.5

    def test_fit_all(self):
        """Test fitting multiple distributions."""
        np.random.seed(42)
        data = np.random.exponential(2, 1000)

        fitter = DistributionFitter(data)
        results = fitter.fit_all(['norm', 'expon', 'gamma'])

        assert len(results) == 3
        assert all('aic' in r for r in results.values())

        best_dist = min(results.items(), key=lambda x: x[1]['aic'])[0]
        assert best_dist in ['expon', 'gamma']

    def test_fit_normal(self):
        """Test normal distribution fitting."""
        np.random.seed(42)
        data = np.random.normal(3, 1.5, 1000)

        fitter = DistributionFitter(data)
        mu, sigma = fitter.fit_normal()

        assert abs(mu - 3) < 0.2
        assert abs(sigma - 1.5) < 0.2

    def test_fit_gamma_mom(self):
        """Test Method of Moments for gamma."""
        np.random.seed(42)
        data = np.random.gamma(2, scale=3, size=1000)

        fitter = DistributionFitter(data)
        shape, scale = fitter.fit_gamma_mom()

        assert abs(shape - 2) < 1
        assert abs(scale - 3) < 1

    def test_qq_plot_data(self):
        """Test Q-Q plot data generation."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        fitter = DistributionFitter(data)
        result = fitter.fit_distribution('norm')

        theoretical, sample = fitter.qq_plot_data('norm', result['parameters'])

        assert len(theoretical) == len(sample)
        assert len(theoretical) == len(data)

    def test_fit_exponential(self):
        """Test exponential distribution fitting."""
        np.random.seed(42)
        data = np.random.exponential(2, 1000)

        fitter = DistributionFitter(data)
        lam = fitter.fit_exponential()

        assert abs(lam - 0.5) < 0.1


class TestBayesianEstimator:
    """Test Bayesian parameter estimation."""

    def test_creation(self):
        """Test estimator creation."""
        data = np.random.normal(0, 1, 100)
        estimator = BayesianEstimator(data)
        assert len(estimator.data) == 100

    def test_estimate_normal_mean(self):
        """Test posterior estimation for normal mean."""
        np.random.seed(42)
        true_mu = 5.0
        data = np.random.normal(true_mu, 2.0, 100)

        estimator = BayesianEstimator(data)
        posterior_mean, posterior_var = estimator.estimate_normal_mean(
            prior_mean=0, prior_var=100, known_variance=4
        )

        assert posterior_var > 0
        assert abs(posterior_mean - true_mu) < 1

    def test_estimate_normal_variance(self):
        """Test posterior estimation for normal variance."""
        np.random.seed(42)
        true_mu = 3.0
        data = np.random.normal(true_mu, 2.0, 100)

        estimator = BayesianEstimator(data)
        posterior_shape, posterior_scale = estimator.estimate_normal_variance(
            prior_shape=1, prior_scale=1, known_mean=true_mu
        )

        posterior_mean_var = posterior_scale / (posterior_shape - 1)
        assert posterior_mean_var > 0

    def test_estimate_bernoulli_p(self):
        """Test posterior for Bernoulli success probability."""
        np.random.seed(42)
        true_p = 0.7
        n = 100
        data = np.random.binomial(1, true_p, n)

        estimator = BayesianEstimator(data)
        posterior_alpha, posterior_beta = estimator.estimate_bernoulli_p(
            prior_alpha=1, prior_beta=1
        )

        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        assert abs(posterior_mean - true_p) < 0.15

    def test_estimate_poisson_rate(self):
        """Test posterior for Poisson rate."""
        np.random.seed(42)
        true_lambda = 5.0
        data = np.random.poisson(true_lambda, 100)

        estimator = BayesianEstimator(data)
        posterior_shape, posterior_rate = estimator.estimate_poisson_rate(
            prior_shape=1, prior_rate=1
        )

        posterior_mean = posterior_shape / posterior_rate
        assert abs(posterior_mean - true_lambda) < 1


class TestGoodnessOfFit:
    """Test goodness-of-fit tests."""

    def test_kolmogorov_smirnov(self):
        """Test Kolmogorov-Smirnov test."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        from scipy import stats
        cdf_func = lambda x: stats.norm.cdf(x, loc=0, scale=1)
        ks_stat, p_value = GoodnessOfFit.kolmogorov_smirnov_test(data, cdf_func)

        assert ks_stat > 0
        assert p_value > 0.05

    def test_anderson_darling(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = GoodnessOfFit.anderson_darling_test(data, 'norm')

        assert 'statistic' in result
        # scipy >= 1.17 returns pvalue, older returns critical_values
        if 'pvalue' in result:
            assert result['pvalue'] > 0.05
        else:
            assert 'critical_values' in result

    def test_chi_square(self):
        """Test Chi-square test."""
        np.random.seed(42)
        observed = np.array([50, 30, 20])
        expected = np.array([40, 40, 20])

        chi2_stat, p_value = GoodnessOfFit.chi_square_test(observed, expected)

        assert chi2_stat > 0
        assert 0 <= p_value <= 1

    def test_shapiro_wilk(self):
        """Test Shapiro-Wilk test for normality."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        w_stat, p_value = GoodnessOfFit.shapiro_wilk_test(data)

        assert 0 <= w_stat <= 1
        assert p_value > 0.05

    def test_shapiro_wilk_non_normal(self):
        """Test Shapiro-Wilk rejects non-normal data."""
        np.random.seed(42)
        data = np.random.exponential(1, 100)

        w_stat, p_value = GoodnessOfFit.shapiro_wilk_test(data)

        assert p_value < 0.05

    def test_jarque_bera(self):
        """Test Jarque-Bera test for normality."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        jb_stat, p_value = GoodnessOfFit.jarque_bera_test(data)

        assert jb_stat > 0
        assert p_value > 0.05


def test_aic_bic_calculation():
    """Test AIC and BIC information criteria."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    fitter = DistributionFitter(data)
    result = fitter.fit_distribution('norm')

    aic = result['aic']
    bic = result['bic']

    assert bic > aic
    assert aic > 0
    assert bic > 0


def test_confidence_intervals():
    """Test Bayesian credible interval for normal mean."""
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000)

    estimator = BayesianEstimator(data)
    posterior_mean, posterior_var = estimator.estimate_normal_mean(
        prior_mean=0, prior_var=100, known_variance=4
    )

    posterior_std = np.sqrt(posterior_var)
    lower = posterior_mean - 1.96 * posterior_std
    upper = posterior_mean + 1.96 * posterior_std

    assert lower < posterior_mean < upper
    assert upper - lower > 0


# ---- Additional edge case and error path tests ----


class TestDistributionFitterEdgeCases:
    def test_creation_with_invalid_data(self):
        with pytest.raises(ValueError, match="Need at least 2"):
            DistributionFitter(np.array([1]))

    def test_fit_all_empty_list(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fitter = DistributionFitter(data)
        results = fitter.fit_all([])
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_fit_beta_valid(self):
        np.random.seed(42)
        data = np.random.beta(2, 5, 1000)
        fitter = DistributionFitter(data)
        alpha, beta_param = fitter.fit_beta()
        assert alpha > 0
        assert beta_param > 0

    def test_fit_beta_invalid_range(self):
        data = np.array([-0.1, 0.5, 1.2])
        fitter = DistributionFitter(data)
        with pytest.raises(ValueError, match="data in \\[0, 1\\]"):
            fitter.fit_beta()

    def test_fit_gamma_mle(self):
        np.random.seed(42)
        data = np.random.gamma(3, scale=2, size=500)
        fitter = DistributionFitter(data)
        shape, scale = fitter.fit_gamma_mle()
        assert shape > 0
        assert scale > 0

    def test_fit_weibull(self):
        np.random.seed(42)
        data = np.random.weibull(1.5, 500) * 2
        fitter = DistributionFitter(data)
        shape, scale = fitter.fit_weibull()
        assert shape > 0
        assert scale > 0

    def test_fit_lognormal_valid(self):
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 500)
        fitter = DistributionFitter(data)
        mu, sigma = fitter.fit_lognormal()
        assert abs(mu) < 1
        assert sigma > 0

    def test_fit_lognormal_invalid_data(self):
        data = np.array([-1, 2, 3])
        fitter = DistributionFitter(data)
        with pytest.raises(ValueError, match="positive data"):
            fitter.fit_lognormal()

    def test_calculate_residuals(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fitter = DistributionFitter(data)
        mu, sigma = fitter.fit_normal()
        residuals = fitter.calculate_residuals('norm', (mu, sigma))
        assert len(residuals) == len(data)
        assert not np.any(np.isnan(residuals))

    def test_calculate_residuals_non_normal(self):
        np.random.seed(42)
        data = np.random.exponential(2, 200)
        fitter = DistributionFitter(data)
        lam = fitter.fit_exponential()
        residuals = fitter.calculate_residuals('expon', (0, 1 / lam))
        assert len(residuals) == len(data)

    def test_fit_distribution_ks_pvalue(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        fitter = DistributionFitter(data)
        result = fitter.fit_distribution('norm')
        assert result['ks_pvalue'] > 0.05

    def test_fit_distribution_on_2d_array(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 2))
        fitter = DistributionFitter(data)
        assert len(fitter.data) == 200

    def test_fit_all_with_invalid_distribution(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fitter = DistributionFitter(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = fitter.fit_all(['norm', 'bogus_dist'])
        assert 'norm' in results
        assert 'bogus_dist' not in results
        assert len(w) >= 1


class TestGoodnessOfFitEdgeCases:
    def test_chi_square_with_df(self):
        observed = np.array([50, 30, 20])
        expected = np.array([40, 40, 20])
        chi2, p = GoodnessOfFit.chi_square_test(observed, expected, df=2)
        assert chi2 > 0
        assert 0 <= p <= 1

    def test_ks_test_non_normal(self):
        np.random.seed(42)
        data = np.random.exponential(1, 200)
        from scipy import stats
        ks_stat, p = GoodnessOfFit.kolmogorov_smirnov_test(
            data, lambda x: stats.norm.cdf(x, loc=1, scale=1)
        )
        assert p < 0.05

    def test_anderson_darling_expon(self):
        np.random.seed(42)
        data = np.random.exponential(1, 500)
        result = GoodnessOfFit.anderson_darling_test(data, 'expon')
        assert 'statistic' in result
        assert ('pvalue' in result or 'critical_values' in result)

    def test_shapiro_wilk_large_warning(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 5001)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GoodnessOfFit.shapiro_wilk_test(data)
        assert len(w) >= 1

    def test_jarque_bera_non_normal(self):
        np.random.seed(42)
        data = np.random.exponential(1, 200)
        jb_stat, p = GoodnessOfFit.jarque_bera_test(data)
        assert p < 0.05
