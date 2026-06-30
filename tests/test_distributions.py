"""Tests for probability distributions."""

import math
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributions import (
    NormalDistribution,
    ExponentialDistribution,
    UniformDistribution,
    BetaDistribution,
    GammaDistribution,
    ChiSquareDistribution,
    StudentTDistribution,
    WeibullDistribution,
    LognormalDistribution,
    CauchyDistribution,
    BinomialDistribution,
    PoissonDistribution,
    GeometricDistribution,
    NegativeBinomialDistribution,
    HypergeometricDistribution,
    DiscreteUniformDistribution,
)


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

class TestNormalDistribution:
    """Test Normal distribution."""

    def test_creation(self):
        dist = NormalDistribution(mu=0, sigma=1)
        assert dist.name == "Normal"
        assert not dist.is_discrete

    def test_default_creation(self):
        dist = NormalDistribution()
        assert dist.mu == 0.0
        assert dist.sigma == 1.0

    def test_parameters(self):
        dist = NormalDistribution()
        params = dist.get_parameters()
        assert params == {"mu": 0.0, "sigma": 1.0}

        dist.set_parameters(mu=5, sigma=2)
        assert dist.mu == 5
        assert dist.sigma == 2
        params = dist.get_parameters()
        assert params["mu"] == 5
        assert params["sigma"] == 2

    def test_parameter_bounds(self):
        dist = NormalDistribution()
        bounds = dist.get_parameter_bounds()
        assert "mu" in bounds
        assert "sigma" in bounds

    def test_pdf(self):
        dist = NormalDistribution(mu=0, sigma=1)
        pdf_at_mean = dist.pdf(np.array([0.0]))
        assert abs(pdf_at_mean[0] - 0.3989) < 0.001

    def test_cdf(self):
        dist = NormalDistribution(mu=0, sigma=1)
        cdf_at_mean = dist.cdf(np.array([0.0]))
        assert abs(cdf_at_mean[0] - 0.5) < 0.001

    def test_statistics(self):
        dist = NormalDistribution(mu=5, sigma=2)
        assert abs(dist.mean() - 5) < 0.001
        assert abs(dist.var() - 4) < 0.001
        assert abs(dist.std() - 2) < 0.001
        assert abs(dist.median() - 5) < 0.001
        assert dist.mode() is not None
        assert abs(dist.skewness() - 0) < 0.001
        assert abs(dist.kurtosis() - 0) < 0.001

    def test_get_statistics(self):
        dist = NormalDistribution(mu=5, sigma=2)
        stats = dist.get_statistics()
        assert abs(stats["mean"] - 5) < 0.001
        assert abs(stats["variance"] - 4) < 0.001
        assert abs(stats["std_dev"] - 2) < 0.001
        assert abs(stats["median"] - 5) < 0.001
        assert stats["mode"] is not None
        assert abs(stats["skewness"] - 0) < 0.001
        assert abs(stats["kurtosis"] - 0) < 0.001
        assert "entropy" in stats

    def test_random_samples(self):
        dist = NormalDistribution(mu=0, sigma=1)
        samples = dist.rvs(size=10000, random_state=42)
        assert len(samples) == 10000
        assert abs(np.mean(samples) - 0) < 0.05
        assert abs(np.std(samples, ddof=0) - 1) < 0.05

    def test_support(self):
        dist = NormalDistribution(mu=0, sigma=1)
        a, b = dist.get_support()
        assert np.isneginf(a)
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            NormalDistribution(mu=0, sigma=-1)

        with pytest.raises(ValueError):
            NormalDistribution(mu=0, sigma=0)


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------

class TestExponentialDistribution:
    """Test Exponential distribution."""

    def test_creation(self):
        dist = ExponentialDistribution(lambda_param=1.0)
        assert dist.name == "Exponential"
        assert not dist.is_discrete

    def test_default_creation(self):
        dist = ExponentialDistribution()
        assert dist.lambda_param == 1.0

    def test_parameters(self):
        dist = ExponentialDistribution(lambda_param=2.0)
        params = dist.get_parameters()
        assert "lambda" in params
        assert params["lambda"] == 2.0

    def test_statistics(self):
        lambda_val = 2.0
        dist = ExponentialDistribution(lambda_param=lambda_val)
        assert abs(dist.mean() - 1 / lambda_val) < 0.001
        assert abs(dist.var() - 1 / (lambda_val**2)) < 0.001
        assert abs(dist.std() - 1 / lambda_val) < 0.001

    def test_pdf_at_zero(self):
        dist = ExponentialDistribution(lambda_param=2.0)
        pdf_at_zero = dist.pdf(np.array([0.0]))
        assert abs(pdf_at_zero[0] - 2.0) < 0.001

    def test_cdf(self):
        dist = ExponentialDistribution(lambda_param=1.0)
        assert abs(dist.cdf(np.array([1.0]))[0] - (1 - np.exp(-1))) < 0.001

    def test_support(self):
        dist = ExponentialDistribution(lambda_param=1.0)
        a, b = dist.get_support()
        assert a == 0.0
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            ExponentialDistribution(lambda_param=0)

        with pytest.raises(ValueError):
            ExponentialDistribution(lambda_param=-1)


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------

class TestUniformDistribution:
    """Test Uniform distribution."""

    def test_creation(self):
        dist = UniformDistribution(a=0, b=1)
        assert dist.name == "Uniform"
        assert not dist.is_discrete

    def test_statistics(self):
        a, b = 2, 8
        dist = UniformDistribution(a=a, b=b)
        assert abs(dist.mean() - (a + b) / 2) < 0.001
        assert abs(dist.var() - (b - a) ** 2 / 12) < 0.001

    def test_pdf_constant(self):
        dist = UniformDistribution(a=0, b=10)
        x = np.linspace(1, 9, 10)
        pdf_vals = dist.pdf(x)
        assert all(abs(p - 0.1) < 0.001 for p in pdf_vals)

    def test_pdf_outside_support(self):
        dist = UniformDistribution(a=0, b=10)
        assert abs(dist.pdf(np.array([-1.0]))[0] - 0.0) < 0.001
        assert abs(dist.pdf(np.array([11.0]))[0] - 0.0) < 0.001

    def test_support(self):
        dist = UniformDistribution(a=2, b=5)
        a, b = dist.get_support()
        assert abs(a - 2) < 0.001
        assert abs(b - 5) < 0.001

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            UniformDistribution(a=5, b=2)

        with pytest.raises(ValueError):
            UniformDistribution(a=5, b=5)


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class TestBetaDistribution:
    """Test Beta distribution."""

    def test_creation(self):
        dist = BetaDistribution(alpha=2, beta=2)
        assert dist.name == "Beta"
        assert not dist.is_discrete

    def test_parameters(self):
        dist = BetaDistribution(alpha=3, beta=4)
        params = dist.get_parameters()
        assert params["alpha"] == 3
        assert params["beta"] == 4

    def test_statistics(self):
        alpha, beta = 2, 5
        dist = BetaDistribution(alpha=alpha, beta=beta)
        expected_mean = alpha / (alpha + beta)
        assert abs(dist.mean() - expected_mean) < 0.001
        expected_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        assert abs(dist.var() - expected_var) < 0.001

    def test_support(self):
        dist = BetaDistribution(alpha=2, beta=2)
        a, b = dist.get_support()
        assert abs(a) < 0.001
        assert abs(b - 1) < 0.001

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            BetaDistribution(alpha=-1, beta=2)

        with pytest.raises(ValueError):
            BetaDistribution(alpha=2, beta=0)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class TestGammaDistribution:
    """Test Gamma distribution."""

    def test_creation(self):
        dist = GammaDistribution(shape=2, scale=2)
        assert dist.name == "Gamma"
        assert not dist.is_discrete

    def test_statistics(self):
        shape, scale = 3, 2
        dist = GammaDistribution(shape=shape, scale=scale)
        assert abs(dist.mean() - shape * scale) < 0.001
        assert abs(dist.var() - shape * scale**2) < 0.001

    def test_support(self):
        dist = GammaDistribution(shape=2, scale=2)
        a, b = dist.get_support()
        assert a == 0.0
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            GammaDistribution(shape=0, scale=2)

        with pytest.raises(ValueError):
            GammaDistribution(shape=2, scale=-1)


# ---------------------------------------------------------------------------
# Chi-Square
# ---------------------------------------------------------------------------

class TestChiSquareDistribution:
    """Test Chi-Square distribution."""

    def test_creation(self):
        dist = ChiSquareDistribution(df=5)
        assert dist.name == "Chi-Square"
        assert not dist.is_discrete

    def test_parameters(self):
        dist = ChiSquareDistribution(df=5)
        params = dist.get_parameters()
        assert params == {"df": 5}

    def test_statistics(self):
        df = 10
        dist = ChiSquareDistribution(df=df)
        assert abs(dist.mean() - df) < 0.001
        assert abs(dist.var() - 2 * df) < 0.001
        assert abs(dist.skewness() - math.sqrt(8 / df)) < 0.01

    def test_support(self):
        dist = ChiSquareDistribution(df=3)
        a, b = dist.get_support()
        assert a == 0.0
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            ChiSquareDistribution(df=0)

        with pytest.raises(ValueError):
            ChiSquareDistribution(df=-2)


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------

class TestStudentTDistribution:
    """Test Student-t distribution."""

    def test_creation(self):
        dist = StudentTDistribution(df=10)
        assert dist.name == "Student-t"
        assert not dist.is_discrete

    def test_statistics(self):
        dist = StudentTDistribution(df=10)
        assert abs(dist.mean() - 0) < 0.001
        assert abs(dist.var() - 10 / 8) < 0.01
        assert abs(dist.skewness() - 0) < 0.001

    def test_support(self):
        dist = StudentTDistribution(df=5)
        a, b = dist.get_support()
        assert np.isneginf(a)
        assert np.isposinf(b)

    def test_df1_undefined_mean(self):
        """Student-t with df=1 (Cauchy-like) has undefined mean and variance."""
        dist = StudentTDistribution(df=1)
        assert not np.isfinite(dist.mean())
        assert not np.isfinite(dist.var())
        assert not np.isfinite(dist.skewness())
        assert not np.isfinite(dist.kurtosis())

    def test_df2_undefined_variance(self):
        """Student-t with df=2 has mean=0 but undefined variance."""
        dist = StudentTDistribution(df=2)
        assert abs(dist.mean() - 0) < 0.001
        assert not np.isfinite(dist.var())
        assert not np.isfinite(dist.skewness())

    def test_df3_defined_moments(self):
        """Student-t with df=3 has defined mean and variance."""
        dist = StudentTDistribution(df=3)
        assert abs(dist.mean() - 0) < 0.001
        assert abs(dist.var() - 3 / 1) < 0.01

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            StudentTDistribution(df=0)

        with pytest.raises(ValueError):
            StudentTDistribution(df=-1)


# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------

class TestWeibullDistribution:
    """Test Weibull distribution."""

    def test_creation(self):
        dist = WeibullDistribution(shape=1.5, scale=1.0)
        assert dist.name == "Weibull"
        assert not dist.is_discrete

    def test_default_creation(self):
        dist = WeibullDistribution()
        assert dist.shape == 1.5
        assert dist.scale == 1.0

    def test_parameters(self):
        dist = WeibullDistribution(shape=2.0, scale=3.0)
        params = dist.get_parameters()
        assert params["shape"] == 2.0
        assert params["scale"] == 3.0

    def test_statistics(self):
        """Weibull with shape=1 is exponential. Test shape=1, scale=2."""
        dist = WeibullDistribution(shape=1.0, scale=2.0)
        assert abs(dist.mean() - 2.0) < 0.001
        assert abs(dist.var() - 4.0) < 0.001

    def test_support(self):
        dist = WeibullDistribution(shape=1.5, scale=1.0)
        a, b = dist.get_support()
        assert a == 0.0
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            WeibullDistribution(shape=0, scale=1)

        with pytest.raises(ValueError):
            WeibullDistribution(shape=1, scale=-1)


# ---------------------------------------------------------------------------
# Lognormal
# ---------------------------------------------------------------------------

class TestLognormalDistribution:
    """Test Lognormal distribution."""

    def test_creation(self):
        dist = LognormalDistribution(mu=0, sigma=1)
        assert dist.name == "Lognormal"
        assert not dist.is_discrete

    def test_default_creation(self):
        dist = LognormalDistribution()
        assert dist.mu == 0.0
        assert dist.sigma == 1.0

    def test_parameters(self):
        dist = LognormalDistribution(mu=1.0, sigma=0.5)
        params = dist.get_parameters()
        assert params["mu"] == 1.0
        assert params["sigma"] == 0.5

    def test_statistics(self):
        mu, sigma = 1.0, 0.5
        dist = LognormalDistribution(mu=mu, sigma=sigma)
        expected_mean = math.exp(mu + sigma**2 / 2)
        expected_var = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
        assert abs(dist.mean() - expected_mean) < 0.001
        assert abs(dist.var() - expected_var) < 0.001

    def test_support(self):
        dist = LognormalDistribution(mu=0, sigma=1)
        a, b = dist.get_support()
        assert a == 0.0
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            LognormalDistribution(mu=0, sigma=0)

        with pytest.raises(ValueError):
            LognormalDistribution(mu=0, sigma=-1)


# ---------------------------------------------------------------------------
# Cauchy
# ---------------------------------------------------------------------------

class TestCauchyDistribution:
    """Test Cauchy distribution."""

    def test_creation(self):
        dist = CauchyDistribution(x0=0, gamma=1)
        assert dist.name == "Cauchy"
        assert not dist.is_discrete

    def test_default_creation(self):
        dist = CauchyDistribution()
        assert dist.x0 == 0.0
        assert dist.gamma == 1.0

    def test_parameters(self):
        dist = CauchyDistribution(x0=2.0, gamma=0.5)
        params = dist.get_parameters()
        assert params["x0"] == 2.0
        assert params["gamma"] == 0.5

    def test_undefined_moments(self):
        """Cauchy distribution has undefined mean, variance, skewness, kurtosis."""
        dist = CauchyDistribution(x0=0, gamma=1)
        assert np.isnan(dist.mean())
        assert np.isnan(dist.var())
        assert np.isnan(dist.skewness())
        assert np.isnan(dist.kurtosis())

    def test_median_equals_x0(self):
        dist = CauchyDistribution(x0=3.0, gamma=1)
        assert abs(dist.median() - 3.0) < 0.001

    def test_mode_equals_x0(self):
        dist = CauchyDistribution(x0=3.0, gamma=1)
        mode_val = dist.mode()
        assert mode_val is not None

    def test_pdf_at_peak(self):
        dist = CauchyDistribution(x0=0, gamma=1)
        pdf_at_peak = dist.pdf(np.array([0.0]))
        expected = 1.0 / math.pi
        assert abs(pdf_at_peak[0] - expected) < 0.001

    def test_cdf_at_median(self):
        dist = CauchyDistribution(x0=0, gamma=1)
        assert abs(dist.cdf(np.array([0.0]))[0] - 0.5) < 0.001

    def test_get_statistics_cauchy(self):
        """get_statistics should handle Cauchy gracefully."""
        dist = CauchyDistribution(x0=0, gamma=1)
        stats = dist.get_statistics()
        assert not np.isfinite(stats["mean"])
        assert not np.isfinite(stats["variance"])
        assert abs(stats["median"] - 0.0) < 0.001
        assert stats["mode"] is not None

    def test_support(self):
        dist = CauchyDistribution(x0=0, gamma=1)
        a, b = dist.get_support()
        assert np.isneginf(a)
        assert np.isposinf(b)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            CauchyDistribution(x0=0, gamma=0)

        with pytest.raises(ValueError):
            CauchyDistribution(x0=0, gamma=-1)


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

class TestBinomialDistribution:
    """Test Binomial distribution."""

    def test_creation(self):
        dist = BinomialDistribution(n=10, p=0.5)
        assert dist.name == "Binomial"
        assert dist.is_discrete

    def test_parameters(self):
        dist = BinomialDistribution(n=10, p=0.5)
        params = dist.get_parameters()
        assert params["n"] == 10
        assert params["p"] == 0.5

    def test_statistics(self):
        n, p = 20, 0.3
        dist = BinomialDistribution(n=n, p=p)
        assert abs(dist.mean() - n * p) < 0.001
        assert abs(dist.var() - n * p * (1 - p)) < 0.001

    def test_pmf_sum(self):
        dist = BinomialDistribution(n=10, p=0.5)
        x = np.arange(0, 11)
        pmf_vals = dist.pdf(x)
        assert abs(np.sum(pmf_vals) - 1.0) < 0.001

    def test_mode(self):
        dist = BinomialDistribution(n=10, p=0.5)
        mode_val = dist.mode()
        assert abs(mode_val - 5) < 0.001

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            BinomialDistribution(n=10, p=1.5)

        with pytest.raises(ValueError):
            BinomialDistribution(n=10, p=-0.1)

        with pytest.raises(ValueError):
            BinomialDistribution(n=-5, p=0.5)

        with pytest.raises(ValueError):
            BinomialDistribution(n=0, p=0.5)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

class TestPoissonDistribution:
    """Test Poisson distribution."""

    def test_creation(self):
        dist = PoissonDistribution(lambda_param=3.0)
        assert dist.name == "Poisson"
        assert dist.is_discrete

    def test_parameters(self):
        dist = PoissonDistribution(lambda_param=5.0)
        params = dist.get_parameters()
        assert "lambda" in params
        assert params["lambda"] == 5.0

    def test_statistics(self):
        lambda_val = 5.0
        dist = PoissonDistribution(lambda_param=lambda_val)
        assert abs(dist.mean() - lambda_val) < 0.001
        assert abs(dist.var() - lambda_val) < 0.001
        assert abs(dist.skewness() - 1 / math.sqrt(lambda_val)) < 0.01

    def test_pmf_values(self):
        dist = PoissonDistribution(lambda_param=2.0)
        pmf_at_zero = dist.pdf(np.array([0]))
        assert abs(pmf_at_zero[0] - math.exp(-2)) < 0.001

    def test_pmf_sum_approximate(self):
        dist = PoissonDistribution(lambda_param=4.0)
        x = np.arange(0, 30)
        pmf_vals = dist.pdf(x)
        assert abs(np.sum(pmf_vals) - 1.0) < 0.01

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            PoissonDistribution(lambda_param=0)

        with pytest.raises(ValueError):
            PoissonDistribution(lambda_param=-3)


# ---------------------------------------------------------------------------
# Geometric
# ---------------------------------------------------------------------------

class TestGeometricDistribution:
    """Test Geometric distribution."""

    def test_creation(self):
        dist = GeometricDistribution(p=0.5)
        assert dist.name == "Geometric"
        assert dist.is_discrete

    def test_statistics(self):
        p = 0.25
        dist = GeometricDistribution(p=p)
        assert abs(dist.mean() - 1 / p) < 0.001
        assert abs(dist.var() - (1 - p) / p**2) < 0.001

    def test_pmf_at_one(self):
        dist = GeometricDistribution(p=0.5)
        pmf_at_one = dist.pdf(np.array([1]))
        assert abs(pmf_at_one[0] - 0.5) < 0.001

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            GeometricDistribution(p=0)

        with pytest.raises(ValueError):
            GeometricDistribution(p=1.5)

        with pytest.raises(ValueError):
            GeometricDistribution(p=-0.1)


# ---------------------------------------------------------------------------
# Negative Binomial
# ---------------------------------------------------------------------------

class TestNegativeBinomialDistribution:
    """Test Negative Binomial distribution."""

    def test_creation(self):
        dist = NegativeBinomialDistribution(r=5, p=0.5)
        assert dist.name == "Negative Binomial"
        assert dist.is_discrete

    def test_default_creation(self):
        dist = NegativeBinomialDistribution()
        assert dist.r == 5
        assert dist.p == 0.5

    def test_parameters(self):
        dist = NegativeBinomialDistribution(r=10, p=0.3)
        params = dist.get_parameters()
        assert params["r"] == 10
        assert params["p"] == 0.3

    def test_statistics(self):
        r, p = 8, 0.4
        dist = NegativeBinomialDistribution(r=r, p=p)
        expected_mean = r * (1 - p) / p
        expected_var = r * (1 - p) / p**2
        assert abs(dist.mean() - expected_mean) < 0.001
        assert abs(dist.var() - expected_var) < 0.001

    def test_pmf_sum_approximate(self):
        dist = NegativeBinomialDistribution(r=3, p=0.5)
        x = np.arange(0, 25)
        pmf_vals = dist.pdf(x)
        assert abs(np.sum(pmf_vals) - 1.0) < 0.01

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            NegativeBinomialDistribution(r=0, p=0.5)

        with pytest.raises(ValueError):
            NegativeBinomialDistribution(r=5, p=0)

        with pytest.raises(ValueError):
            NegativeBinomialDistribution(r=5, p=1.5)


# ---------------------------------------------------------------------------
# Hypergeometric
# ---------------------------------------------------------------------------

class TestHypergeometricDistribution:
    """Test Hypergeometric distribution."""

    def test_creation(self):
        dist = HypergeometricDistribution(M=20, n=7, N=12)
        assert dist.name == "Hypergeometric"
        assert dist.is_discrete

    def test_parameters(self):
        dist = HypergeometricDistribution(M=20, n=7, N=12)
        params = dist.get_parameters()
        assert params["M"] == 20
        assert params["n"] == 7
        assert params["N"] == 12

    def test_statistics(self):
        M, n, N = 20, 7, 12
        dist = HypergeometricDistribution(M=M, n=n, N=N)
        expected_mean = N * n / M
        expected_var = N * n / M * (M - n) / M * (M - N) / (M - 1)
        assert abs(dist.mean() - expected_mean) < 0.001
        assert abs(dist.var() - expected_var) < 0.001

    def test_pmf_sum(self):
        dist = HypergeometricDistribution(M=15, n=5, N=8)
        low, high = dist.get_support()
        x = np.arange(int(low), int(high) + 1)
        pmf_vals = dist.pdf(x)
        assert abs(np.sum(pmf_vals) - 1.0) < 0.001

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            HypergeometricDistribution(M=20, n=25, N=12)

        with pytest.raises(ValueError):
            HypergeometricDistribution(M=20, n=7, N=25)

        with pytest.raises(ValueError):
            HypergeometricDistribution(M=-5, n=7, N=12)

        with pytest.raises(ValueError):
            HypergeometricDistribution(M=20, n=-1, N=12)


# ---------------------------------------------------------------------------
# Discrete Uniform
# ---------------------------------------------------------------------------

class TestDiscreteUniformDistribution:
    """Test Discrete Uniform distribution."""

    def test_creation(self):
        dist = DiscreteUniformDistribution(low=1, high=6)
        assert dist.name == "Discrete Uniform"
        assert dist.is_discrete

    def test_default_creation(self):
        dist = DiscreteUniformDistribution()
        assert dist.low == 1
        assert dist.high == 6

    def test_parameters(self):
        dist = DiscreteUniformDistribution(low=0, high=10)
        params = dist.get_parameters()
        assert params["low"] == 0
        assert params["high"] == 10

    def test_statistics(self):
        low, high = 1, 6
        dist = DiscreteUniformDistribution(low=low, high=high)
        n = high - low + 1
        expected_mean = (low + high) / 2
        expected_var = (n**2 - 1) / 12
        assert abs(dist.mean() - expected_mean) < 0.001
        assert abs(dist.var() - expected_var) < 0.001

    def test_pmf_constant(self):
        dist = DiscreteUniformDistribution(low=1, high=6)
        x = np.arange(1, 7)
        pmf_vals = dist.pdf(x)
        assert all(abs(p - 1 / 6) < 0.001 for p in pmf_vals)

    def test_pmf_sum(self):
        dist = DiscreteUniformDistribution(low=1, high=6)
        x = np.arange(1, 7)
        pmf_vals = dist.pdf(x)
        assert abs(np.sum(pmf_vals) - 1.0) < 0.001

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            DiscreteUniformDistribution(low=5, high=5)

        with pytest.raises(ValueError):
            DiscreteUniformDistribution(low=10, high=5)


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------

def test_distribution_string_representation():
    """Test string representation."""
    dist = NormalDistribution(mu=5, sigma=2)
    repr_str = repr(dist)
    assert "Normal" in repr_str
    assert "mu=5" in repr_str
    assert "sigma=2" in repr_str


def test_quantile_functions():
    """Test quantile functions (ppf is inverse of cdf)."""
    dist = NormalDistribution(mu=0, sigma=1)
    x = 1.5
    q = dist.cdf(np.array([x]))[0]
    x_recovered = dist.ppf(np.array([q]))[0]
    assert abs(x - x_recovered) < 0.001


def test_confidence_intervals():
    """Test confidence interval calculation."""
    dist = NormalDistribution(mu=0, sigma=1)
    lower, upper = dist.interval(0.95)
    assert abs(lower - (-1.96)) < 0.01
    assert abs(upper - 1.96) < 0.01


def test_get_statistics_comprehensive():
    """Test get_statistics returns all expected keys."""
    dist = NormalDistribution(mu=5, sigma=2)
    stats = dist.get_statistics()

    expected_keys = {"mean", "variance", "std_dev", "median",
                     "mode", "skewness", "kurtosis", "entropy"}
    assert set(stats.keys()) == expected_keys
    assert abs(stats["mean"] - 5) < 0.001
    assert abs(stats["variance"] - 4) < 0.001
    assert abs(stats["std_dev"] - 2) < 0.001
    assert abs(stats["median"] - 5) < 0.001
    assert stats["mode"] is not None


def test_all_16_distributions_can_be_created():
    """Sanity check that all 16 distributions instantiate without error."""
    distributions = [
        NormalDistribution(),
        ExponentialDistribution(),
        UniformDistribution(),
        BetaDistribution(),
        GammaDistribution(),
        ChiSquareDistribution(),
        StudentTDistribution(),
        WeibullDistribution(),
        LognormalDistribution(),
        CauchyDistribution(),
        BinomialDistribution(),
        PoissonDistribution(),
        GeometricDistribution(),
        NegativeBinomialDistribution(),
        HypergeometricDistribution(),
        DiscreteUniformDistribution(),
    ]
    for dist in distributions:
        assert dist.name is not None
        params = dist.get_parameters()
        assert len(params) > 0


# ===========================================================================
# Tests for uninitialized distribution error paths (base.py coverage)
# ===========================================================================

class TestUninitializedDistributionErrors:
    """Test error paths when distribution is not initialized."""

    def test_uninitialized_mode(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.mode()

    def test_uninitialized_skewness(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.skewness()

    def test_uninitialized_kurtosis(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.kurtosis()

    def test_uninitialized_entropy(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.entropy()

    def test_uninitialized_get_support(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.get_support()

    def test_uninitialized_interval(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.interval(0.95)

    def test_uninitialized_pdf(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.pdf(np.array([0.0]))

    def test_uninitialized_cdf(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.cdf(np.array([0.0]))

    def test_uninitialized_ppf(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.ppf(np.array([0.5]))

    def test_uninitialized_rvs(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.rvs(size=5)

    def test_uninitialized_mean(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.mean()

    def test_uninitialized_var(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.var()

    def test_uninitialized_std(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.std()

    def test_uninitialized_median(self):
        dist = NormalDistribution.__new__(NormalDistribution)
        dist.name = "Test"
        dist.is_discrete = False
        dist._dist = None
        with pytest.raises(ValueError, match="not initialized"):
            dist.median()


# ===========================================================================
# Tests for get_parameter_bounds on all continuous distributions
# ===========================================================================

class TestParameterBoundsContinuous:
    def test_exponential_bounds(self):
        dist = ExponentialDistribution()
        bounds = dist.get_parameter_bounds()
        assert "lambda" in bounds
        low, high = bounds["lambda"]
        assert low > 0
        assert low < high

    def test_uniform_bounds(self):
        dist = UniformDistribution()
        bounds = dist.get_parameter_bounds()
        assert "a" in bounds
        assert "b" in bounds
        a_low, a_high = bounds["a"]
        b_low, b_high = bounds["b"]
        assert a_low < a_high
        assert b_low < b_high

    def test_beta_bounds(self):
        dist = BetaDistribution()
        bounds = dist.get_parameter_bounds()
        assert "alpha" in bounds
        assert "beta" in bounds
        assert bounds["alpha"][0] > 0

    def test_gamma_bounds(self):
        dist = GammaDistribution()
        bounds = dist.get_parameter_bounds()
        assert "shape" in bounds
        assert "scale" in bounds
        assert bounds["shape"][0] > 0

    def test_chisquare_bounds(self):
        dist = ChiSquareDistribution()
        bounds = dist.get_parameter_bounds()
        assert "df" in bounds
        low, high = bounds["df"]
        assert low >= 1

    def test_studentt_bounds(self):
        dist = StudentTDistribution()
        bounds = dist.get_parameter_bounds()
        assert "df" in bounds
        low, high = bounds["df"]
        assert low > 0

    def test_weibull_bounds(self):
        dist = WeibullDistribution()
        bounds = dist.get_parameter_bounds()
        assert "shape" in bounds
        assert "scale" in bounds

    def test_lognormal_bounds(self):
        dist = LognormalDistribution()
        bounds = dist.get_parameter_bounds()
        assert "mu" in bounds
        assert "sigma" in bounds
        assert bounds["sigma"][0] > 0

    def test_cauchy_bounds(self):
        dist = CauchyDistribution()
        bounds = dist.get_parameter_bounds()
        assert "x0" in bounds
        assert "gamma" in bounds
        assert bounds["gamma"][0] > 0


# ===========================================================================
# Tests for mode() covering the inner exception catch path  
# ===========================================================================

class TestModeCoverage:
    """Test mode() for distributions where dist.mode() raises an exception."""

    def test_exponential_mode_via_density(self):
        dist = ExponentialDistribution(lambda_param=2.0)
        result = dist.mode()
        assert abs(result - 0.0) < 0.01

    def test_lognormal_mode(self):
        dist = LognormalDistribution(mu=0, sigma=0.5)
        result = dist.mode()
        assert result > 0

    def test_cauchy_mode(self):
        dist = CauchyDistribution(x0=3.0, gamma=1)
        result = dist.mode()
        assert result is not None
        assert np.isfinite(result)

    def test_discrete_uniform_mode(self):
        dist = DiscreteUniformDistribution(low=1, high=6)
        result = dist.mode()
        assert 1 <= result <= 6

    def test_beta_mode(self):
        dist = BetaDistribution(alpha=3, beta=2)
        result = dist.mode()
        assert 0 < result < 1

    def test_get_statistics_mode_exception_handling(self):
        dist = CauchyDistribution(x0=0, gamma=1)
        stats = dist.get_statistics()
        assert stats["mode"] is not None


# ===========================================================================
# Additional edge-case tests
# ===========================================================================

class TestAdditionalEdgeCases:
    def test_entropy_normal(self):
        dist = NormalDistribution(mu=0, sigma=1)
        ent = dist.entropy()
        assert ent > 0

    def test_kurtosis_normal(self):
        dist = NormalDistribution(mu=5, sigma=2)
        k = dist.kurtosis()
        assert abs(k) < 0.01

    def test_set_parameters_partial(self):
        dist = NormalDistribution(mu=0, sigma=1)
        dist.set_parameters(mu=10)
        assert dist.mu == 10
        assert dist.sigma == 1

    def test_interval_custom_alpha(self):
        dist = NormalDistribution(mu=0, sigma=1)
        lower, upper = dist.interval(0.99)
        assert lower < upper
        width_99 = upper - lower
        lower95, upper95 = dist.interval(0.95)
        width_95 = upper95 - lower95
        assert width_99 > width_95

    def test_repr_discrete(self):
        dist = BinomialDistribution(n=10, p=0.5)
        r = repr(dist)
        assert "Binomial" in r
        assert "n=10" in r

    def test_ppf_bounds(self):
        dist = NormalDistribution(mu=0, sigma=1)
        lower = dist.ppf(np.array([0.025]))[0]
        upper = dist.ppf(np.array([0.975]))[0]
        assert lower < 0 < upper

    def test_rvs_multiple_sizes(self):
        dist = NormalDistribution(mu=0, sigma=1)
        s1 = dist.rvs(size=1, random_state=42)
        s10 = dist.rvs(size=10, random_state=42)
        assert isinstance(s1, np.ndarray)
        assert len(s10) == 10
