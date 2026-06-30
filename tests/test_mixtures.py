"""Tests for mixture distributions."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributions.mixtures import (
    MixtureDistribution,
    GaussianMixtureModel,
    BayesianGMM,
    select_optimal_components,
)
from distributions.continuous import NormalDistribution


class TestMixtureDistribution:
    """Test general mixture distribution."""

    def test_creation(self):
        """Test mixture creation."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=5, sigma=1)
        weights = [0.3, 0.7]

        mixture = MixtureDistribution([comp1, comp2], weights)
        assert mixture.n_components == 2
        assert np.allclose(mixture.weights, weights)

    def test_invalid_weights_sum(self):
        """Test that weights not summing to 1 raises error."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=5, sigma=1)
        weights = [0.3, 0.5]

        with pytest.raises(ValueError):
            MixtureDistribution([comp1, comp2], weights)

    def test_invalid_weights_length(self):
        """Test that mismatched components and weights raises error."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=5, sigma=1)
        weights = [1.0]

        with pytest.raises(ValueError):
            MixtureDistribution([comp1, comp2], weights)

    def test_pdf(self):
        """Test PDF calculation."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=5, sigma=1)
        weights = [0.5, 0.5]

        mixture = MixtureDistribution([comp1, comp2], weights)

        x = np.array([0, 5])
        pdf_vals = mixture.pdf(x)

        assert all(pdf_vals > 0)
        assert abs(pdf_vals[0] - pdf_vals[1]) < 0.1

    def test_cdf(self):
        """Test CDF calculation."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=5, sigma=1)
        weights = [0.5, 0.5]

        mixture = MixtureDistribution([comp1, comp2], weights)

        x = np.array([-5, 0, 2.5, 5, 10])
        cdf_vals = mixture.cdf(x)

        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))
        assert all(0 <= c <= 1 for c in cdf_vals)

    def test_random_samples(self):
        """Test random sample generation."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=10, sigma=1)
        weights = [0.7, 0.3]

        mixture = MixtureDistribution([comp1, comp2], weights)

        samples = mixture.rvs(size=10000, random_state=42)
        assert len(samples) == 10000

        samples_low = samples[samples < 5]
        samples_high = samples[samples >= 5]

        ratio = len(samples_low) / len(samples)
        assert abs(ratio - 0.7) < 0.05

    def test_mean(self):
        """Test mean calculation."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=10, sigma=1)
        weights = [0.5, 0.5]

        mixture = MixtureDistribution([comp1, comp2], weights)

        expected_mean = 0.5 * 0 + 0.5 * 10
        assert abs(mixture.mean() - expected_mean) < 0.001


class TestGaussianMixtureModel:
    """Test Gaussian Mixture Model."""

    def test_creation(self):
        """Test GMM creation."""
        gmm = GaussianMixtureModel(n_components=2)
        assert gmm.n_components == 2

    def test_fit(self):
        """Test GMM fitting."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(5, 1, 500)
        data = np.concatenate([data1, data2]).reshape(-1, 1)

        gmm = GaussianMixtureModel(n_components=2)
        gmm.fit(data)

        assert gmm.fitted
        params = gmm.get_parameters()
        assert len(params['means']) == 2
        assert len(params['covariances']) == 2
        assert len(params['weights']) == 2
        assert abs(np.sum(params['weights']) - 1.0) < 0.01

    def test_predict(self):
        """Test cluster prediction."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(10, 1, 500)
        data = np.concatenate([data1, data2]).reshape(-1, 1)

        gmm = GaussianMixtureModel(n_components=2)
        gmm.fit(data)

        labels = gmm.predict(data)
        assert len(labels) == len(data)
        assert set(labels) == {0, 1}

    def test_score_samples(self):
        """Test log-likelihood scoring."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(5, 1, 500)
        data = np.concatenate([data1, data2]).reshape(-1, 1)

        gmm = GaussianMixtureModel(n_components=2)
        gmm.fit(data)

        scores = gmm.score_samples(data)
        assert len(scores) == len(data)

    def test_aic_bic(self):
        """Test AIC and BIC calculation."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500).reshape(-1, 1)

        gmm = GaussianMixtureModel(n_components=2)
        gmm.fit(data)

        aic = gmm.aic(data)
        bic = gmm.bic(data)

        assert bic > aic


class TestBayesianGMM:
    """Test Bayesian Gaussian Mixture Model."""

    def test_creation(self):
        """Test Bayesian GMM creation."""
        bgmm = BayesianGMM(max_components=3)
        assert bgmm.max_components == 3

    def test_fit(self):
        """Test Bayesian GMM fitting."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 300)
        data2 = np.random.normal(5, 1, 300)
        data = np.concatenate([data1, data2]).reshape(-1, 1)

        bgmm = BayesianGMM(max_components=3)
        bgmm.fit(data)

        assert bgmm.fitted
        active = bgmm.get_active_components()
        assert active >= 1

    def test_automatic_component_pruning(self):
        """Test that Bayesian GMM prunes unnecessary components."""
        import warnings
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(5, 1, 500)
        data = np.concatenate([data1, data2]).reshape(-1, 1)

        bgmm = BayesianGMM(max_components=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bgmm.fit(data)

        active_components = bgmm.get_active_components()
        assert active_components < 5


def test_select_optimal_components():
    """Test optimal component selection."""
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 500)
    data2 = np.random.normal(5, 1, 500)
    data = np.concatenate([data1, data2]).reshape(-1, 1)

    optimal_n, results = select_optimal_components(data, max_components=5)

    assert 'optimal_components' in results
    assert 'aic_scores' in results
    assert 'bic_scores' in results
    assert 'components_range' in results

    assert optimal_n in [2, 3]
    assert results['optimal_components'] == optimal_n

    assert len(results['aic_scores']) == 5
    assert len(results['bic_scores']) == 5


def test_mixture_fit_em():
    """Test EM algorithm for mixture fitting."""
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 500)
    data2 = np.random.normal(10, 2, 500)
    data = np.concatenate([data1, data2])

    comp1 = NormalDistribution(mu=0, sigma=1)
    comp2 = NormalDistribution(mu=5, sigma=1)
    mixture = MixtureDistribution([comp1, comp2], [0.5, 0.5])

    responsibilities, components, weights = mixture.fit_em(
        data, n_components=2, max_iter=100
    )

    assert responsibilities.shape == (len(data), 2)
    assert len(components) == 2
    assert len(weights) == 2
    assert abs(np.sum(weights) - 1.0) < 0.01

    means = [comp.mean() for comp in components]
    means_sorted = sorted(means)
    assert abs(means_sorted[0] - 0) < 2
    assert abs(means_sorted[1] - 10) < 2


# =============================================================================
# Additional error path and edge-case tests
# =============================================================================

class TestMixtureErrorPaths:
    """Test MixtureDistribution error paths."""

    def test_negative_weights(self):
        """Test that negative weights raise error."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=5, sigma=1)
        with pytest.raises(ValueError, match="non-negative"):
            MixtureDistribution([comp1, comp2], [1.2, -0.2])

    def test_var(self):
        """Test variance calculation."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=10, sigma=1)
        mixture = MixtureDistribution([comp1, comp2], [0.5, 0.5])
        v = mixture.var()
        assert v > 0
        assert np.isfinite(v)

    def test_mean(self):
        """Test mean of mixture."""
        comp1 = NormalDistribution(mu=0, sigma=1)
        comp2 = NormalDistribution(mu=10, sigma=1)
        mixture = MixtureDistribution([comp1, comp2], [0.3, 0.7])
        expected = 0.3 * 0 + 0.7 * 10
        assert abs(mixture.mean() - expected) < 0.001


class TestGMMErrorPaths:
    """Test GaussianMixtureModel error paths."""

    def test_predict_before_fit(self):
        """Test that predict raises before fit."""
        gmm = GaussianMixtureModel(n_components=2)
        data = np.random.normal(0, 1, 10).reshape(-1, 1)
        with pytest.raises(ValueError, match="fitted"):
            gmm.predict(data)

    def test_sample_before_fit(self):
        """Test that sample/rvs raises before fit."""
        gmm = GaussianMixtureModel(n_components=2)
        with pytest.raises(ValueError, match="fitted"):
            gmm.rvs(size=5)

    def test_score_samples_before_fit(self):
        """Test that score_samples raises before fit."""
        gmm = GaussianMixtureModel(n_components=2)
        data = np.random.normal(0, 1, 10).reshape(-1, 1)
        with pytest.raises(ValueError, match="fitted"):
            gmm.score_samples(data)

    def test_bic_before_fit(self):
        """Test that bic raises before fit."""
        gmm = GaussianMixtureModel(n_components=2)
        data = np.random.normal(0, 1, 10).reshape(-1, 1)
        with pytest.raises(ValueError, match="fitted"):
            gmm.bic(data)

    def test_aic_before_fit(self):
        """Test that aic raises before fit."""
        gmm = GaussianMixtureModel(n_components=2)
        data = np.random.normal(0, 1, 10).reshape(-1, 1)
        with pytest.raises(ValueError, match="fitted"):
            gmm.aic(data)

    def test_get_parameters_before_fit(self):
        """Test that get_parameters raises before fit."""
        gmm = GaussianMixtureModel(n_components=2)
        with pytest.raises(ValueError, match="fitted"):
            gmm.get_parameters()

    def test_pdf(self):
        """Test PDF after fit."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).reshape(-1, 1)
        gmm = GaussianMixtureModel(n_components=2)
        gmm.fit(data)
        pdf_vals = gmm.pdf(np.array([[0.0], [1.0]]))
        assert len(pdf_vals) == 2
        assert np.all(pdf_vals > 0)


class TestBayesianGMMErrorPaths:
    """Test BayesianGMM error paths."""

    def test_predict_before_fit(self):
        """Test that predict raises before fit."""
        bgmm = BayesianGMM(max_components=3)
        data = np.random.normal(0, 1, 10).reshape(-1, 1)
        with pytest.raises(ValueError, match="fitted"):
            bgmm.predict(data)

    def test_get_active_components_before_fit(self):
        """Test that get_active_components raises before fit."""
        bgmm = BayesianGMM(max_components=3)
        with pytest.raises(ValueError, match="fitted"):
            bgmm.get_active_components()

    def test_get_active_components_after_fit(self):
        """Test get_active_components returns value after fit."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 300)
        data2 = np.random.normal(5, 1, 300)
        data = np.concatenate([data1, data2]).reshape(-1, 1)
        bgmm = BayesianGMM(max_components=3)
        bgmm.fit(data)
        active = bgmm.get_active_components()
        assert active >= 1
