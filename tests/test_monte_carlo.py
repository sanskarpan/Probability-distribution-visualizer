"""Tests for Monte Carlo simulation."""

import warnings
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monte_carlo.simulator import (
    MonteCarloSimulator,
    SimulationResult,
    VarianceReduction,
    QuasiMonteCarloSimulator,
)


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation functionality."""

    def test_creation(self):
        """Test simulator creation."""
        sim = MonteCarloSimulator(random_seed=42)
        assert sim.random_seed == 42

    def test_estimate_expectation(self):
        """Test expectation estimation."""
        sim = MonteCarloSimulator(random_seed=42)

        def sampler():
            return np.random.normal(5, 2)

        estimate, std_error = sim.estimate_expectation(sampler, num_samples=10000)

        assert abs(estimate - 5) < 0.2
        assert std_error > 0

    def test_estimate_probability(self):
        """Test probability estimation."""

        sim = MonteCarloSimulator(random_seed=42)

        def event_func():
            x = np.random.normal(0, 1)
            return x > 0

        result = sim.estimate_probability(event_func, num_samples=10000)

        assert abs(result['probability'] - 0.5) < 0.05
        assert 'confidence_interval' in result
        assert result['num_samples'] == 10000
        assert result['num_successes'] > 0

    def test_importance_sampling(self):
        """Test importance sampling."""
        sim = MonteCarloSimulator(random_seed=42)

        def proposal_sampler():
            return np.random.normal(3, 1)

        def proposal_pdf(x):
            from scipy.stats import norm
            return norm.pdf(x, 3, 1)

        def target_pdf(x):
            from scipy.stats import norm
            return norm.pdf(x, 0, 1)

        def target_func(x):
            return 1 if x > 3 else 0

        estimate, std_error = sim.importance_sampling(
            target_func, proposal_sampler, proposal_pdf, target_pdf,
            num_samples=10000
        )

        assert abs(estimate - 0.00135) < 0.01
        assert std_error > 0

    def test_stratified_sampling(self):
        """Test stratified sampling."""
        sim = MonteCarloSimulator(random_seed=42)

        def func(x):
            return x ** 2

        strata_bounds = [(i / 10, (i + 1) / 10) for i in range(10)]
        result = sim.stratified_sampling(func, strata_bounds, num_samples_per_stratum=100)

        assert abs(result.mean - 1 / 3) < 0.01
        assert result.std > 0

    def test_antithetic_variates(self):
        """Test antithetic variates variance reduction."""

        def sampler():
            return np.random.uniform(0, 1)

        def func(x):
            return x ** 2

        estimate, std_error = VarianceReduction.antithetic_variates(sampler, func, num_pairs=5000)

        assert abs(estimate - 1 / 3) < 0.05
        assert std_error > 0

    def test_bootstrap(self):
        """Test bootstrap resampling."""
        sim = MonteCarloSimulator(random_seed=42)

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        def statistic(sample):
            return np.mean(sample)

        result = sim.bootstrap(data, statistic, num_bootstrap=1000)

        assert abs(result['estimate'] - 10) < 0.5
        assert 'bootstrap_std' in result
        assert 'confidence_interval' in result
        assert 'bootstrap_distribution' in result

    def test_permutation_test(self):
        """Test permutation test."""
        sim = MonteCarloSimulator(random_seed=42)

        np.random.seed(42)
        group1 = np.random.normal(5, 1, 50)
        group2 = np.random.normal(5, 1, 50)

        def test_statistic(g1, g2):
            return abs(np.mean(g1) - np.mean(g2))

        result = sim.permutation_test(group1, group2, test_statistic, num_permutations=1000)

        assert 'observed_statistic' in result
        assert 'p_value' in result
        assert 'permutation_distribution' in result

    def test_permutation_test_different_groups(self):
        """Test permutation test with different groups."""
        sim = MonteCarloSimulator(random_seed=42)

        np.random.seed(42)
        group1 = np.random.normal(5, 1, 100)
        group2 = np.random.normal(7, 1, 100)

        def test_statistic(g1, g2):
            return abs(np.mean(g1) - np.mean(g2))

        result = sim.permutation_test(group1, group2, test_statistic, num_permutations=1000)

        assert result['p_value'] < 0.05


class TestVarianceReduction:
    """Test variance reduction techniques."""

    def test_control_variates(self):
        """Test control variates method."""
        np.random.seed(42)

        def target_sampler():
            return np.random.normal(0, 1)

        def target_func(x):
            return np.exp(x)

        def control_func(x):
            return x

        estimate, std_error = VarianceReduction.control_variates(
            target_sampler, target_func, control_func, control_mean=0, num_samples=10000
        )

        assert isinstance(estimate, float)
        assert std_error > 0


class TestQuasiMonteCarloSimulator:
    """Test Quasi-Monte Carlo simulation."""

    def test_creation(self):
        """Test QMC simulator creation."""
        qmc = QuasiMonteCarloSimulator()
        assert isinstance(qmc, QuasiMonteCarloSimulator)

    def test_halton_sequence(self):
        """Test Halton sequence generation."""
        samples = QuasiMonteCarloSimulator.halton_sequence(n=100, base=2)

        assert len(samples) == 100
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_sobol_sequence(self):
        """Test Sobol sequence generation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = QuasiMonteCarloSimulator.sobol_sequence(n=128, dim=3)

        assert samples.shape == (128, 3)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_estimate_integral(self):
        """Test integral estimation with QMC."""
        qmc = QuasiMonteCarloSimulator()

        def func(x, y):
            return x * y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = qmc.integrate_qmc(func, bounds=[(0, 1), (0, 1)], num_points=8192)

        assert abs(result - 0.25) < 0.01

    def test_qmc_vs_mc_convergence(self):
        """Test that QMC converges faster than MC."""

        def func(x):
            return x ** 2

        qmc = QuasiMonteCarloSimulator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qmc_result = qmc.integrate_qmc(func, bounds=[(0, 1)], num_points=1024)
        qmc_error = abs(qmc_result - 1 / 3)

        mc = MonteCarloSimulator(random_seed=42)
        mc_samples = np.random.uniform(0, 1, (1000, 1))
        mc_values = np.array([func(s[0]) for s in mc_samples])
        mc_estimate = np.mean(mc_values)
        mc_error = abs(mc_estimate - 1 / 3)

        assert qmc_error < mc_error * 1.5

    def test_transform_to_distribution(self):
        """Test transforming uniform samples to target distribution."""
        uniform_samples = QuasiMonteCarloSimulator.halton_sequence(n=1000, base=2)

        from scipy.stats import norm
        normal_samples = norm.ppf(uniform_samples)

        assert abs(np.mean(normal_samples) - 0) < 0.2
        assert abs(np.std(normal_samples) - 1) < 0.2


def test_simulation_result():
    """Test SimulationResult data class."""
    result = SimulationResult(
        mean=5.0,
        std=0.1,
        var=0.01,
        median=5.0,
        quantiles={0.5: 5.0},
        samples=np.array([5.0]),
        confidence_interval=(4.8, 5.2),
    )

    assert result.mean == 5.0
    assert result.std == 0.1
    assert result.var == 0.01
    assert result.confidence_interval[0] == 4.8
    assert result.confidence_interval[1] == 5.2


def test_integration_estimate_pi():
    """Test estimating π using Monte Carlo."""
    sim = MonteCarloSimulator(random_seed=42)

    def event_func():
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        return x ** 2 + y ** 2 <= 1

    result = sim.estimate_probability(event_func, num_samples=100000)

    pi_estimate = result['probability'] * 4
    assert abs(pi_estimate - np.pi) < 0.05


def test_confidence_interval_coverage():
    """Test that confidence intervals have correct coverage."""
    sim = MonteCarloSimulator(random_seed=42)

    n_simulations = 100
    coverage_count = 0
    true_mean = 10

    for i in range(n_simulations):
        def sampler():
            return np.random.normal(true_mean, 2)

        result = sim.simulate(sampler, num_samples=1000)
        ci = result.confidence_interval

        if ci[0] <= true_mean <= ci[1]:
            coverage_count += 1

    coverage = coverage_count / n_simulations

    assert 0.85 <= coverage <= 1.0


def test_simulate_basic():
    sim = MonteCarloSimulator(random_seed=42)

    def sampler():
        return np.random.normal(10, 2)

    result = sim.simulate(sampler, num_samples=1000)
    assert abs(result.mean - 10) < 0.3
    assert result.std > 0
    assert result.var > 0
    assert hasattr(result, 'confidence_interval')
    assert result.confidence_interval[0] < result.confidence_interval[1]


def test_simulate_with_convergence():
    sim = MonteCarloSimulator(random_seed=42)

    def sampler():
        return np.random.exponential(2)

    result = sim.simulate(sampler, num_samples=500, track_convergence=True)
    assert result.convergence_data is not None
    assert len(result.convergence_data) == 500


def test_simulate_custom_confidence():
    sim = MonteCarloSimulator(random_seed=42)

    def sampler():
        return np.random.normal(0, 1)

    result = sim.simulate(sampler, num_samples=1000, confidence_level=0.99)
    ci_width = result.confidence_interval[1] - result.confidence_interval[0]
    assert ci_width > 0


def test_simulate_quantiles():
    sim = MonteCarloSimulator(random_seed=42)

    def sampler():
        return np.random.normal(0, 1)

    result = sim.simulate(sampler, num_samples=10000)
    assert 0.01 in result.quantiles
    assert 0.05 in result.quantiles
    assert 0.50 in result.quantiles
    assert 0.95 in result.quantiles
    assert 0.99 in result.quantiles
    assert abs(result.quantiles[0.50]) < 0.2


def test_estimate_expectation_kwargs():
    sim = MonteCarloSimulator(random_seed=42)

    def sampler(mean=0, sd=1):
        return np.random.normal(mean, sd)

    est, se = sim.estimate_expectation(
        lambda mean=0, sd=1: np.random.normal(mean, sd),
        num_samples=5000
    )
    assert abs(est) < 0.2


def test_importance_sampling_zero_variance():
    sim = MonteCarloSimulator(random_seed=42)

    def proposal_sampler():
        return np.random.normal(0, 1)

    from scipy.stats import norm

    def proposal_pdf(x):
        return norm.pdf(x, 0, 1)

    def target_pdf(x):
        return norm.pdf(x, 0, 1)

    def target_func(x):
        return x

    est, se = sim.importance_sampling(
        target_func, proposal_sampler, proposal_pdf, target_pdf,
        num_samples=5000
    )
    assert abs(est) < 0.2


def test_stratified_sampling_asymmetric():
    sim = MonteCarloSimulator(random_seed=42)

    def func(x):
        return np.exp(x)

    strata_bounds = [(i / 5, (i + 1) / 5) for i in range(5)]
    result = sim.stratified_sampling(func, strata_bounds, num_samples_per_stratum=200)
    actual = np.exp(1) - 1
    assert abs(result.mean - actual) < 0.1


def test_bootstrap_median():
    sim = MonteCarloSimulator(random_seed=42)
    np.random.seed(42)
    data = np.random.normal(10, 2, 200)

    def statistic(sample):
        return np.median(sample)

    result = sim.bootstrap(data, statistic, num_bootstrap=500, confidence_level=0.90)
    assert 'estimate' in result
    assert 'bootstrap_mean' in result
    assert 'confidence_interval' in result
    assert result['confidence_interval'][0] < result['confidence_interval'][1]


def test_permutation_test_tight_threshold():
    sim = MonteCarloSimulator(random_seed=42)
    np.random.seed(42)
    group1 = np.random.normal(5, 1, 30)
    group2 = np.random.normal(10, 1, 30)

    def test_statistic(g1, g2):
        return np.mean(g1) - np.mean(g2)

    result = sim.permutation_test(group1, group2, test_statistic, num_permutations=2000)
    assert result['p_value'] < 0.05


def test_estimate_probability_edge():
    sim = MonteCarloSimulator(random_seed=42)

    def event_func():
        return True

    result = sim.estimate_probability(event_func, num_samples=1000)
    assert np.isclose(result['probability'], 1.0)


def test_estimate_probability_impossible():
    sim = MonteCarloSimulator(random_seed=42)

    def event_func():
        return False

    result = sim.estimate_probability(event_func, num_samples=1000)
    assert np.isclose(result['probability'], 0.0)


class TestVarianceReductionEdgeCases:
    def test_antithetic_variates_cosine(self):
        def sampler():
            return np.random.uniform(0, 2 * np.pi)

        def func(x):
            return np.cos(x)

        est, se = VarianceReduction.antithetic_variates(sampler, func, num_pairs=3000)
        assert abs(est) < 0.1

    def test_control_variates_high_correlation(self):
        np.random.seed(42)

        def target_sampler():
            return np.random.normal(0, 1)

        def target_func(x):
            return x ** 3

        def control_func(x):
            return x

        est, se = VarianceReduction.control_variates(
            target_sampler, target_func, control_func, control_mean=0, num_samples=5000
        )
        assert abs(est) < 0.5

    def test_control_variates_no_correlation(self):
        np.random.seed(42)

        def target_sampler():
            return np.random.normal(0, 1)

        def target_func(x):
            return x

        def control_func(x):
            return np.random.normal(0, 1)

        est, se = VarianceReduction.control_variates(
            target_sampler, target_func, control_func, control_mean=0, num_samples=5000
        )
        assert abs(est) < 0.2


class TestSimulationResult:
    def test_with_convergence(self):
        conv_data = np.cumsum(np.random.normal(0, 1, 100)) / np.arange(1, 101)
        result = SimulationResult(
            mean=0.1, std=1.0, var=1.0, median=0.05,
            quantiles={0.5: 0.05}, samples=np.random.normal(0, 1, 100),
            confidence_interval=(-2.0, 2.0), convergence_data=conv_data
        )
        assert result.convergence_data is not None
        assert len(result.convergence_data) == 100

    def test_full_result_properties(self):
        samples = np.random.normal(5, 2, 1000)
        result = SimulationResult(
            mean=np.mean(samples),
            std=np.std(samples),
            var=np.var(samples),
            median=np.median(samples),
            quantiles={0.25: np.percentile(samples, 25), 0.75: np.percentile(samples, 75)},
            samples=samples,
            confidence_interval=(3.0, 7.0),
        )
        assert result.mean > 0
        assert result.median > 0
        assert len(result.samples) == 1000