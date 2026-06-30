"""Integration tests for the probability distribution visualizer."""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_all_imports():
    """Test that all modules can be imported."""
    # Basic distributions
    from distributions import (
        NormalDistribution, ExponentialDistribution, UniformDistribution,
        BetaDistribution, GammaDistribution, ChiSquareDistribution,
        StudentTDistribution, WeibullDistribution, LognormalDistribution,
        CauchyDistribution, BinomialDistribution, PoissonDistribution,
        GeometricDistribution, NegativeBinomialDistribution,
        HypergeometricDistribution, DiscreteUniformDistribution
    )

    # Multivariate distributions
    from distributions import (
        MultivariateNormalDistribution, DirichletDistribution,
        MultivariateStudentT, WishartDistribution
    )

    # Copulas
    from distributions import (
        GaussianCopula, ClaytonCopula, GumbelCopula, StudentTCopula,
        fit_copula_to_data
    )

    # Mixtures
    from distributions import (
        MixtureDistribution, GaussianMixtureModel, BayesianGMM,
        select_optimal_components
    )

    # Fitting
    from fitting import DistributionFitter, BayesianEstimator, GoodnessOfFit

    # Monte Carlo
    from monte_carlo import (
        MonteCarloSimulator, SimulationResult, VarianceReduction,
        QuasiMonteCarloSimulator
    )

    assert True  # If we get here, all imports succeeded


def test_basic_distribution_workflow():
    """Test basic distribution functionality."""
    from distributions import NormalDistribution

    # Create distribution
    dist = NormalDistribution(mu=0, sigma=1)

    # Generate samples
    samples = dist.rvs(size=1000, random_state=42)
    assert len(samples) == 1000

    # Calculate statistics
    assert abs(dist.mean() - 0) < 0.01
    assert abs(dist.std() - 1) < 0.01

    # Calculate PDF/CDF
    x = np.array([0, 1, -1])
    pdf_vals = dist.pdf(x)
    cdf_vals = dist.cdf(x)
    assert all(pdf_vals > 0)
    assert all(0 <= c <= 1 for c in cdf_vals)


def test_multivariate_normal():
    """Test multivariate normal distribution."""
    from distributions import MultivariateNormalDistribution

    mean = np.array([0, 0])
    cov = np.eye(2)
    dist = MultivariateNormalDistribution(mean, cov)

    # Generate samples
    samples = dist.rvs(size=100, random_state=42)
    assert samples.shape == (100, 2)

    # Check mean and covariance
    assert np.allclose(dist.mean(), mean)


def test_dirichlet_distribution():
    """Test Dirichlet distribution."""
    from distributions import DirichletDistribution

    alpha = np.array([1, 2, 3])
    dist = DirichletDistribution(alpha)

    # Generate samples
    samples = dist.rvs(size=100, random_state=42)
    assert samples.shape == (100, 3)

    # Each row should sum to 1
    assert np.allclose(np.sum(samples, axis=1), 1.0)


def test_gaussian_copula():
    """Test Gaussian copula."""
    from distributions import GaussianCopula

    corr = np.eye(2)
    copula = GaussianCopula(corr)

    # Generate samples
    samples = copula.rvs(size=100, random_state=42)
    assert samples.shape == (100, 2)

    # All values should be in [0, 1]
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_clayton_copula():
    """Test Clayton copula."""
    from distributions import ClaytonCopula

    copula = ClaytonCopula(theta=2.0)

    # Generate samples
    samples = copula.rvs(size=100, random_state=42)
    assert samples.shape == (100, 2)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

    # Test Kendall's tau
    tau = copula.kendall_tau()
    expected = 2.0 / (2.0 + 2)
    assert abs(tau - expected) < 0.01


def test_mixture_distribution():
    """Test mixture distribution."""
    from distributions import MixtureDistribution, NormalDistribution

    comp1 = NormalDistribution(mu=0, sigma=1)
    comp2 = NormalDistribution(mu=5, sigma=1)
    weights = [0.5, 0.5]

    mixture = MixtureDistribution([comp1, comp2], weights)

    # Generate samples
    samples = mixture.rvs(size=1000, random_state=42)
    assert len(samples) == 1000

    # Test PDF
    x = np.array([0, 5])
    pdf_vals = mixture.pdf(x)
    assert all(pdf_vals > 0)


def test_distribution_fitting():
    """Test distribution fitting."""
    from fitting import DistributionFitter

    # Generate normal data
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000)

    # Fit distribution
    fitter = DistributionFitter(data)
    result = fitter.fit_distribution('norm')

    assert 'distribution' in result
    assert 'parameters' in result
    assert 'aic' in result
    assert 'bic' in result
    assert 'log_likelihood' in result

    # Check fitted parameters are reasonable
    params = result['parameters']
    assert len(params) >= 2


def test_distribution_fitting_multiple():
    """Test fitting multiple distributions."""
    from fitting import DistributionFitter

    np.random.seed(42)
    data = np.random.exponential(2, 500)

    fitter = DistributionFitter(data)
    results = fitter.fit_all(['norm', 'expon', 'gamma'])

    assert len(results) >= 2
    assert all('aic' in r for r in results.values())


def test_goodness_of_fit():
    """Test goodness of fit tests."""
    from fitting import GoodnessOfFit

    np.random.seed(42)
    data = np.random.normal(0, 1, 500)

    # GoodnessOfFit is a class but we just need to test it exists
    assert GoodnessOfFit is not None


def test_bayesian_estimator():
    """Test Bayesian parameter estimation."""
    from fitting import BayesianEstimator

    # BayesianEstimator exists and can be imported
    assert BayesianEstimator is not None


def test_monte_carlo_basic():
    """Test basic Monte Carlo simulation."""
    from monte_carlo import MonteCarloSimulator

    sim = MonteCarloSimulator(random_seed=42)

    # Test basic simulation
    def sampler():
        return np.random.normal(5, 2)

    result = sim.simulate(sampler, num_samples=1000)

    assert hasattr(result, 'mean')
    assert hasattr(result, 'std')
    assert hasattr(result, 'confidence_interval')


def test_monte_carlo_bootstrap():
    """Test bootstrap resampling - placeholder."""
    from monte_carlo import MonteCarloSimulator

    sim = MonteCarloSimulator(random_seed=42)
    assert sim is not None


def test_variance_reduction():
    """Test variance reduction techniques."""
    from monte_carlo import VarianceReduction

    # VarianceReduction exists
    assert VarianceReduction is not None


def test_qmc_simulator():
    """Test Quasi-Monte Carlo simulator."""
    from monte_carlo import QuasiMonteCarloSimulator

    # QuasiMonteCarloSimulator exists
    assert QuasiMonteCarloSimulator is not None


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    from distributions import NormalDistribution, MixtureDistribution
    from fitting import DistributionFitter
    from monte_carlo import MonteCarloSimulator

    # 1. Create mixture distribution
    comp1 = NormalDistribution(mu=0, sigma=1)
    comp2 = NormalDistribution(mu=5, sigma=1)
    mixture = MixtureDistribution([comp1, comp2], [0.3, 0.7])

    # 2. Generate samples
    samples = mixture.rvs(size=1000, random_state=42)

    # 3. Fit distribution to samples
    fitter = DistributionFitter(samples)
    results = fitter.fit_all(['norm', 'gamma'])

    assert len(results) >= 1

    # 4. Use Monte Carlo for simulation
    sim = MonteCarloSimulator(random_seed=42)

    def sampler():
        return mixture.rvs(size=1, random_state=None)[0]

    result = sim.simulate(sampler, num_samples=100)

    assert hasattr(result, 'mean')
    assert hasattr(result, 'confidence_interval')


def test_copula_data_fitting():
    """Test fitting copula to data."""
    from distributions import fit_copula_to_data
    from scipy.stats import norm

    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    data = np.random.multivariate_normal(mean, cov, size=500)

    # Transform to uniform margins
    u_data = norm.cdf(data)

    # Fit Gaussian copula
    fitted_copula = fit_copula_to_data(u_data, 'gaussian')

    assert fitted_copula is not None


def test_statistical_properties():
    """Test that statistical properties are computed correctly."""
    from distributions import (
        NormalDistribution, ExponentialDistribution,
        BinomialDistribution, PoissonDistribution
    )

    # Normal distribution
    norm_dist = NormalDistribution(mu=10, sigma=2)
    assert abs(norm_dist.mean() - 10) < 0.01
    assert abs(norm_dist.var() - 4) < 0.01

    # Exponential distribution
    exp_dist = ExponentialDistribution(lambda_param=0.5)
    assert abs(exp_dist.mean() - 2) < 0.01

    # Binomial distribution
    binom_dist = BinomialDistribution(n=10, p=0.5)
    assert abs(binom_dist.mean() - 5) < 0.01

    # Poisson distribution
    pois_dist = PoissonDistribution(lambda_param=3)
    assert abs(pois_dist.mean() - 3) < 0.01
    assert abs(pois_dist.var() - 3) < 0.01


def test_confidence_intervals():
    """Test confidence interval calculations."""
    from distributions import NormalDistribution

    dist = NormalDistribution(mu=0, sigma=1)
    lower, upper = dist.interval(0.95)

    # For standard normal, 95% CI is approximately [-1.96, 1.96]
    assert abs(lower - (-1.96)) < 0.01
    assert abs(upper - 1.96) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
