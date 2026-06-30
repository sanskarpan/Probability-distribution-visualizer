"""Tests for copula functions."""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributions.copulas import (
    Copula,
    GaussianCopula,
    ClaytonCopula,
    GumbelCopula,
    StudentTCopula,
    fit_copula_to_data,
)


class TestGaussianCopula:
    """Test Gaussian copula."""

    def test_creation(self):
        """Test copula creation."""
        corr = np.array([[1, 0.5], [0.5, 1]])
        copula = GaussianCopula(corr)
        assert copula.dimension == 2
        assert np.allclose(copula.correlation, corr)

    def test_invalid_correlation(self):
        """Test that invalid correlation raises error."""
        # Not positive definite
        corr = np.array([[1, 2], [2, 1]])
        with pytest.raises(ValueError):
            GaussianCopula(corr)

    def test_diagonal_correlation(self):
        """Test that diagonal must be 1."""
        corr = np.array([[0.9, 0.5], [0.5, 1]])
        with pytest.raises(ValueError):
            GaussianCopula(corr)

    def test_random_samples(self):
        """Test random sample generation."""
        corr = np.eye(2)
        copula = GaussianCopula(corr)

        samples = copula.rvs(size=1000, random_state=42)
        assert samples.shape == (1000, 2)

        # All values should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_pdf(self):
        """Test PDF calculation."""
        corr = np.array([[1, 0.5], [0.5, 1]])
        copula = GaussianCopula(corr)

        u = np.array([[0.5, 0.5]])
        pdf = copula.pdf(u)
        assert pdf[0] > 0

    def test_independence(self):
        """Test that identity correlation gives independence."""
        corr = np.eye(2)
        copula = GaussianCopula(corr)

        samples = copula.rvs(size=10000, random_state=42)

        # For independent copula, correlation should be near 0
        sample_corr = np.corrcoef(samples.T)[0, 1]
        assert abs(sample_corr) < 0.1

    def test_kendall_tau(self):
        """Test Kendall's tau calculation."""
        corr = np.array([[1, 0.5], [0.5, 1]])
        copula = GaussianCopula(corr)
        tau = copula.kendall_tau()

        # For Gaussian copula: tau ≈ (2/π) * arcsin(ρ)
        expected = (2 / np.pi) * np.arcsin(0.5)
        assert abs(tau - expected) < 0.01


class TestClaytonCopula:
    """Test Clayton copula."""

    def test_creation(self):
        """Test copula creation."""
        copula = ClaytonCopula(theta=2.0)
        assert copula.dimension == 2
        assert copula.theta == 2.0

    def test_invalid_theta(self):
        """Test that invalid theta raises error."""
        with pytest.raises(ValueError):
            ClaytonCopula(theta=-2.0)  # Must be > -1

    def test_random_samples(self):
        """Test random sample generation."""
        copula = ClaytonCopula(theta=2.0)

        samples = copula.rvs(size=1000, random_state=42)
        assert samples.shape == (1000, 2)

        # All values should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_pdf(self):
        """Test PDF calculation."""
        copula = ClaytonCopula(theta=2.0)

        u = np.array([[0.5, 0.5]])
        pdf = copula.pdf(u)
        assert pdf[0] > 0

    def test_kendall_tau(self):
        """Test Kendall's tau calculation."""
        theta = 2.0
        copula = ClaytonCopula(theta=theta)
        tau = copula.kendall_tau()

        # For Clayton: tau = θ/(θ+2)
        expected = theta / (theta + 2)
        assert abs(tau - expected) < 0.01

    def test_lower_tail_dependence(self):
        """Test that Clayton has lower tail dependence."""
        copula = ClaytonCopula(theta=2.0)
        samples = copula.rvs(size=10000, random_state=42)

        # Check for lower tail dependence
        lower_tail = samples[samples[:, 0] < 0.1]
        if len(lower_tail) > 0:
            # In lower tail, u and v should be correlated
            assert np.mean(lower_tail[:, 1]) < 0.5


class TestGumbelCopula:
    """Test Gumbel copula."""

    def test_creation(self):
        """Test copula creation."""
        copula = GumbelCopula(theta=2.0)
        assert copula.dimension == 2
        assert copula.theta == 2.0

    def test_invalid_theta(self):
        """Test that theta < 1 raises error."""
        with pytest.raises(ValueError):
            GumbelCopula(theta=0.5)

    def test_random_samples(self):
        """Test random sample generation."""
        copula = GumbelCopula(theta=2.0)

        samples = copula.rvs(size=1000, random_state=42)
        assert samples.shape == (1000, 2)

        # All values should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_kendall_tau(self):
        """Test Kendall's tau calculation."""
        theta = 2.0
        copula = GumbelCopula(theta=theta)
        tau = copula.kendall_tau()

        # For Gumbel: tau = (θ-1)/θ
        expected = (theta - 1) / theta
        assert abs(tau - expected) < 0.01

    def test_upper_tail_dependence(self):
        """Test that Gumbel has upper tail dependence."""
        copula = GumbelCopula(theta=3.0)
        samples = copula.rvs(size=10000, random_state=42)

        # Check for upper tail dependence
        upper_tail = samples[samples[:, 0] > 0.9]
        if len(upper_tail) > 0:
            # In upper tail, u and v should be correlated
            assert np.mean(upper_tail[:, 1]) > 0.5


class TestStudentTCopula:
    """Test Student-t copula."""

    def test_creation(self):
        """Test copula creation."""
        corr = np.array([[1, 0.5], [0.5, 1]])
        copula = StudentTCopula(corr, df=5)
        assert copula.dimension == 2
        assert copula.df == 5

    def test_invalid_df(self):
        """Test that invalid df raises error."""
        corr = np.eye(2)
        with pytest.raises(ValueError):
            StudentTCopula(corr, df=0)

    def test_random_samples(self):
        """Test random sample generation."""
        corr = np.eye(2)
        copula = StudentTCopula(corr, df=5)

        samples = copula.rvs(size=1000, random_state=42)
        assert samples.shape == (1000, 2)

        # All values should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_tail_dependence(self):
        """Test that Student-t has both tail dependencies."""
        corr = np.array([[1, 0.7], [0.7, 1]])
        copula = StudentTCopula(corr, df=3)

        samples = copula.rvs(size=10000, random_state=42)

        # Check for tail dependence
        lower_tail = samples[samples[:, 0] < 0.05]
        upper_tail = samples[samples[:, 0] > 0.95]

        if len(lower_tail) > 10 and len(upper_tail) > 10:
            # Both tails should show dependence
            assert np.std(lower_tail[:, 1]) > 0
            assert np.std(upper_tail[:, 1]) > 0


def test_fit_copula_to_data():
    """Test copula fitting to data."""
    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]
    data = np.random.multivariate_normal(mean, cov, size=1000)

    # Transform to uniform margins
    from scipy.stats import norm
    u_data = norm.cdf(data)

    # Fit Gaussian copula
    fitted_copula = fit_copula_to_data(u_data, 'gaussian')
    assert isinstance(fitted_copula, GaussianCopula)

    # Fitted correlation should be close to original
    fitted_corr = fitted_copula.correlation[0, 1]
    assert abs(fitted_corr - 0.7) < 0.2

    # Fit Clayton copula
    fitted_clayton = fit_copula_to_data(u_data, 'clayton')
    assert isinstance(fitted_clayton, ClaytonCopula)
    assert fitted_clayton.theta > 0

    # Fit Gumbel copula
    fitted_gumbel = fit_copula_to_data(u_data, 'gumbel')
    assert isinstance(fitted_gumbel, GumbelCopula)
    assert fitted_gumbel.theta >= 1


def test_copula_transforms():
    """Test that copula samples have uniform margins."""
    corr = np.array([[1, 0.5], [0.5, 1]])
    copula = GaussianCopula(corr)

    samples = copula.rvs(size=10000, random_state=42)

    # Test uniformity using Kolmogorov-Smirnov test
    from scipy.stats import kstest
    for i in range(2):
        _, p_value = kstest(samples[:, i], 'uniform')
        assert p_value > 0.01  # Should not reject uniformity


# ===========================================================================
# CDF tests for copulas
# ===========================================================================

class TestCopulaCDF:
    def test_gaussian_copula_cdf(self):
        corr = np.array([[1, 0.5], [0.5, 1]])
        copula = GaussianCopula(corr)
        u = np.array([[0.5, 0.5]])
        cdf_vals = copula.cdf(u)
        assert cdf_vals[0] > 0
        assert cdf_vals[0] < 1

    def test_clayton_copula_cdf(self):
        copula = ClaytonCopula(theta=2.0)
        u = np.array([[0.5, 0.5]])
        cdf_vals = copula.cdf(u)
        assert cdf_vals[0] > 0
        assert cdf_vals[0] < 1

    def test_gumbel_copula_cdf(self):
        copula = GumbelCopula(theta=2.0)
        u = np.array([[0.5, 0.5]])
        cdf_vals = copula.cdf(u)
        assert cdf_vals[0] > 0
        assert cdf_vals[0] < 1

    def test_clayton_cdf_monotonic(self):
        copula = ClaytonCopula(theta=1.5)
        u1 = np.array([[0.3, 0.3]])
        u2 = np.array([[0.7, 0.7]])
        assert copula.cdf(u1)[0] < copula.cdf(u2)[0]

    def test_gumbel_cdf_monotonic(self):
        copula = GumbelCopula(theta=2.0)
        u1 = np.array([[0.3, 0.3]])
        u2 = np.array([[0.7, 0.7]])
        assert copula.cdf(u1)[0] < copula.cdf(u2)[0]

    def test_gaussian_cdf_monotonic(self):
        corr = np.array([[1, 0.5], [0.5, 1]])
        copula = GaussianCopula(corr)
        u1 = np.array([[0.3, 0.3]])
        u2 = np.array([[0.7, 0.7]])
        assert copula.cdf(u1)[0] < copula.cdf(u2)[0]

    def test_clayton_cdf_multiple_points(self):
        copula = ClaytonCopula(theta=2.0)
        u = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        cdf_vals = copula.cdf(u)
        assert len(cdf_vals) == 3
        assert np.all(np.diff(cdf_vals) > 0)

    def test_gumbel_cdf_multiple_points(self):
        copula = GumbelCopula(theta=2.0)
        u = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        cdf_vals = copula.cdf(u)
        assert len(cdf_vals) == 3
        assert np.all(np.diff(cdf_vals) > 0)


# ===========================================================================
# Error path and edge case tests
# ===========================================================================

class TestCopulaErrorPaths:
    def test_clayton_pdf_not_implemented_beyond_bivariate(self):
        copula = ClaytonCopula(theta=2.0, dimension=3)
        u = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(NotImplementedError, match="bivariate"):
            copula.pdf(u)

    def test_clayton_pdf_higher_dim_not_impl(self):
        copula = ClaytonCopula(theta=2.0, dimension=3)
        u = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(NotImplementedError, match="bivariate"):
            copula.pdf(u)

    def test_clayton_rvs_higher_dim_not_impl(self):
        copula = ClaytonCopula(theta=2.0, dimension=3)
        with pytest.raises(NotImplementedError, match="bivariate"):
            copula.rvs(size=10)

    def test_gumbel_pdf_higher_dim_not_impl(self):
        copula = GumbelCopula(theta=2.0, dimension=3)
        u = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(NotImplementedError, match="bivariate"):
            copula.pdf(u)

    def test_gumbel_rvs_higher_dim_not_impl(self):
        copula = GumbelCopula(theta=2.0, dimension=3)
        with pytest.raises(NotImplementedError, match="bivariate"):
            copula.rvs(size=10)

    def test_gaussian_copula_invalid_square_matrix(self):
        with pytest.raises(ValueError, match="correlation must be square"):
            GaussianCopula(np.array([1, 2, 3]))

    def test_student_t_copula_invalid_square_matrix(self):
        with pytest.raises(ValueError, match="correlation must be square"):
            StudentTCopula(np.array([1, 2, 3]), df=5)

    def test_base_copula_cdf_not_implemented(self):
        base = Copula.__new__(Copula)
        base.name = "Test"
        base.dimension = 2
        with pytest.raises(NotImplementedError):
            base.cdf(np.array([[0.5, 0.5]]))

    def test_base_copula_pdf_not_implemented(self):
        base = Copula.__new__(Copula)
        base.name = "Test"
        base.dimension = 2
        with pytest.raises(NotImplementedError):
            base.pdf(np.array([[0.5, 0.5]]))

    def test_base_copula_rvs_not_implemented(self):
        base = Copula.__new__(Copula)
        base.name = "Test"
        base.dimension = 2
        with pytest.raises(NotImplementedError):
            base.rvs(size=10)

    def test_student_t_copula_kendall_tau_dimension(self):
        corr = np.eye(3)
        copula = StudentTCopula(corr, df=5)
        with pytest.raises(ValueError, match="bivariate"):
            copula.kendall_tau()

    def test_student_t_copula_repr(self):
        corr = np.eye(2)
        copula = StudentTCopula(corr, df=5)
        r = repr(copula)
        assert "StudentTCopula" in r

    def test_gaussian_kendall_tau_non_bivariate(self):
        corr = np.eye(3)
        copula = GaussianCopula(corr)
        with pytest.raises(ValueError, match="bivariate"):
            copula.kendall_tau()

    def test_fit_unsupported_copula_type(self):
        np.random.seed(42)
        data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 500)
        with pytest.raises(ValueError, match="Unsupported copula type"):
            from distributions.copulas import fit_copula_to_data
            fit_copula_to_data(data, copula_type='unsupported_type')

    def test_fit_t_copula(self):
        np.random.seed(42)
        data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 500)
        from distributions.copulas import fit_copula_to_data
        result = fit_copula_to_data(data, copula_type='t')
        assert isinstance(result, StudentTCopula)
        assert result.df == 4


def test_gumbel_copula_pdf_validity():
    copula = GumbelCopula(theta=2.0)
    u = np.array([[0.5, 0.5]])
    pdf_val = copula.pdf(u)
    assert pdf_val[0] > 0
    assert np.isfinite(pdf_val[0])


def test_clayton_copula_theta_zero():
    with pytest.raises(ValueError, match="theta must be non-zero"):
        ClaytonCopula(theta=0.0)


def test_student_t_copula_invalid_df_negative():
    corr = np.eye(2)
    with pytest.raises(ValueError, match="df must be positive"):
        StudentTCopula(corr, df=-1)
