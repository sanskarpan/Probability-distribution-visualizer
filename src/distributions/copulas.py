"""Copulas for modeling dependencies between random variables."""

from typing import Tuple, Optional
import numpy as np
from scipy import stats
from scipy.special import gamma
from scipy.optimize import brentq


class Copula:
    """Base class for copulas."""

    def __init__(self, name: str, dimension: int = 2):
        """
        Initialize copula.

        Args:
            name: Copula name
            dimension: Number of dimensions
        """
        self.name = name
        self.dimension = dimension

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Copula CDF.

        Args:
            u: Uniform(0,1) marginals (shape: n x d or d)

        Returns:
            Copula CDF values
        """
        raise NotImplementedError

    def pdf(self, u: np.ndarray) -> np.ndarray:
        """
        Copula density.

        Args:
            u: Uniform(0,1) marginals

        Returns:
            Copula density values
        """
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from copula.

        Args:
            size: Number of samples
            random_state: Random seed

        Returns:
            Uniform(0,1) samples (shape: size x d)
        """
        raise NotImplementedError

    def kendall_tau(self) -> float:
        """Calculate Kendall's tau."""
        raise NotImplementedError


class GaussianCopula(Copula):
    """Gaussian (Normal) copula."""

    def __init__(self, correlation: np.ndarray):
        """
        Initialize Gaussian copula.

        Args:
            correlation: Correlation matrix (must be positive definite)
        """
        corr = np.asarray(correlation)

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("correlation must be square matrix")

        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("diagonal of correlation matrix must be 1")

        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            raise ValueError("correlation must be positive definite")

        super().__init__("Gaussian", corr.shape[0])
        self.correlation = corr

        # Create multivariate normal for sampling
        self._mvn = stats.multivariate_normal(
            mean=np.zeros(self.dimension),
            cov=corr
        )

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """Gaussian copula CDF."""
        u = np.atleast_2d(u)

        # Transform uniforms to standard normals
        z = stats.norm.ppf(u)

        # Evaluate multivariate normal CDF
        # Note: This is computationally expensive for high dimensions
        cdf_vals = np.array([self._mvn.cdf(zi) for zi in z])

        return cdf_vals

    def pdf(self, u: np.ndarray) -> np.ndarray:
        """Gaussian copula density."""
        u = np.atleast_2d(u)

        # Transform to standard normals
        z = stats.norm.ppf(u)

        # Correlation matrix determinant and inverse
        corr_det = np.linalg.det(self.correlation)
        corr_inv = np.linalg.inv(self.correlation)

        # Copula density
        # c(u) = |Σ|^(-1/2) * exp(z^T (Σ^(-1) - I) z / 2)
        # where z = Φ^(-1)(u)

        identity = np.eye(self.dimension)
        diff = corr_inv - identity

        pdf_vals = np.zeros(len(z))
        for i, zi in enumerate(z):
            exponent = -0.5 * zi @ diff @ zi
            pdf_vals[i] = (1 / np.sqrt(corr_det)) * np.exp(exponent)

        return pdf_vals

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from Gaussian copula."""
        if random_state is not None:
            np.random.seed(random_state)

        # Sample from multivariate normal
        z = self._mvn.rvs(size=size)
        if size == 1:
            z = z.reshape(1, -1)

        # Transform to uniform
        u = stats.norm.cdf(z)

        return u

    def kendall_tau(self) -> float:
        """
        Calculate Kendall's tau (for bivariate case).

        Returns:
            Kendall's tau
        """
        if self.dimension != 2:
            raise ValueError("Kendall's tau only defined for bivariate copula")

        rho = self.correlation[0, 1]
        return (2 / np.pi) * np.arcsin(rho)

    def __repr__(self) -> str:
        return f"GaussianCopula(dimension={self.dimension})"


class ClaytonCopula(Copula):
    """Clayton copula (Archimedean)."""

    def __init__(self, theta: float, dimension: int = 2):
        """
        Initialize Clayton copula.

        Args:
            theta: Dependence parameter (theta >= -1/(d-1), theta != 0)
            dimension: Number of dimensions
        """
        if theta == 0:
            raise ValueError("theta must be non-zero")

        min_theta = -1.0 / (dimension - 1) if dimension > 1 else -np.inf
        if theta < min_theta:
            raise ValueError(f"theta must be >= {min_theta}")

        super().__init__("Clayton", dimension)
        self.theta = theta

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """Clayton copula CDF."""
        u = np.atleast_2d(u)

        # C(u1, ..., ud) = (u1^(-θ) + ... + ud^(-θ) - d + 1)^(-1/θ)
        sum_terms = np.sum(u ** (-self.theta), axis=1)
        cdf_vals = (sum_terms - self.dimension + 1) ** (-1 / self.theta)

        return cdf_vals

    def pdf(self, u: np.ndarray) -> np.ndarray:
        """Clayton copula density (bivariate only)."""
        if self.dimension != 2:
            raise NotImplementedError("PDF only implemented for bivariate")

        u = np.atleast_2d(u)
        u1, u2 = u[:, 0], u[:, 1]

        # c(u1, u2) = (1 + θ) * (u1*u2)^(-1-θ) * (u1^(-θ) + u2^(-θ) - 1)^(-2-1/θ)
        theta = self.theta

        term1 = (1 + theta)
        term2 = (u1 * u2) ** (-1 - theta)
        term3 = (u1 ** (-theta) + u2 ** (-theta) - 1) ** (-2 - 1/theta)

        pdf_vals = term1 * term2 * term3

        return pdf_vals

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from Clayton copula (bivariate only)."""
        if self.dimension != 2:
            raise NotImplementedError("Sampling only implemented for bivariate")

        if random_state is not None:
            np.random.seed(random_state)

        # Algorithm: Use conditional distribution method
        u1 = np.random.uniform(0, 1, size)
        v = np.random.uniform(0, 1, size)

        # u2 = (u1^(-θ) * (v^(-θ/(1+θ)) - 1) + 1)^(-1/θ)
        theta = self.theta
        u2 = (u1 ** (-theta) * (v ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)

        return np.column_stack([u1, u2])

    def kendall_tau(self) -> float:
        """
        Calculate Kendall's tau.

        Returns:
            Kendall's tau
        """
        return self.theta / (self.theta + 2)

    def __repr__(self) -> str:
        return f"ClaytonCopula(theta={self.theta}, dimension={self.dimension})"


class GumbelCopula(Copula):
    """Gumbel copula (Archimedean)."""

    def __init__(self, theta: float, dimension: int = 2):
        """
        Initialize Gumbel copula.

        Args:
            theta: Dependence parameter (theta >= 1)
            dimension: Number of dimensions
        """
        if theta < 1:
            raise ValueError("theta must be >= 1")

        super().__init__("Gumbel", dimension)
        self.theta = theta

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """Gumbel copula CDF."""
        u = np.atleast_2d(u)

        # C(u1, ..., ud) = exp(-(sum_i (-log(ui))^θ)^(1/θ))
        log_terms = (-np.log(u)) ** self.theta
        sum_terms = np.sum(log_terms, axis=1)
        cdf_vals = np.exp(-(sum_terms ** (1 / self.theta)))

        return cdf_vals

    def pdf(self, u: np.ndarray) -> np.ndarray:
        """Gumbel copula density (bivariate only)."""
        if self.dimension != 2:
            raise NotImplementedError("PDF only implemented for bivariate")

        u = np.atleast_2d(u)
        u1, u2 = u[:, 0], u[:, 1]

        theta = self.theta

        # Complex formula for Gumbel copula density
        log_u1 = -np.log(u1)
        log_u2 = -np.log(u2)

        A = (log_u1 ** theta + log_u2 ** theta) ** (1/theta)
        B = (log_u1 ** theta + log_u2 ** theta) ** (-2 + 2/theta)
        C = (log_u1 * log_u2) ** (theta - 1)
        D = 1 + (theta - 1) * (log_u1 ** theta + log_u2 ** theta) ** (-1/theta)

        pdf_vals = np.exp(-A) * B * C * D / (u1 * u2)

        return pdf_vals

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from Gumbel copula (bivariate only)."""
        if self.dimension != 2:
            raise NotImplementedError("Sampling only implemented for bivariate")

        if random_state is not None:
            np.random.seed(random_state)

        theta = self.theta
        u1 = np.random.uniform(0, 1, size)
        p = np.random.uniform(0, 1, size)
        u2 = np.zeros(size)

        for i in range(size):
            u1_i = max(u1[i], 1e-15)
            t = -np.log(u1_i)
            p_i = p[i]

            def cond_cdf(v):
                if v <= 1e-15:
                    return 0.0
                if v >= 1 - 1e-15:
                    return 1.0
                s = -np.log(v)
                A = (t ** theta + s ** theta) ** (1.0 / theta)
                return np.exp(-A) / u1_i * (t / A) ** (theta - 1)

            try:
                u2[i] = brentq(lambda v: cond_cdf(v) - p_i, 1e-15, 1 - 1e-15)
            except Exception:
                u2[i] = 0.5

        return np.column_stack([u1, u2])

    def kendall_tau(self) -> float:
        """
        Calculate Kendall's tau.

        Returns:
            Kendall's tau
        """
        return 1 - 1 / self.theta

    def __repr__(self) -> str:
        return f"GumbelCopula(theta={self.theta}, dimension={self.dimension})"


class StudentTCopula(Copula):
    """Student-t copula."""

    def __init__(self, correlation: np.ndarray, df: float):
        """
        Initialize Student-t copula.

        Args:
            correlation: Correlation matrix
            df: Degrees of freedom
        """
        corr = np.asarray(correlation)

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("correlation must be square matrix")

        if df <= 0:
            raise ValueError("df must be positive")

        super().__init__("Student-t", corr.shape[0])
        self.correlation = corr
        self.df = df

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from Student-t copula."""
        if random_state is not None:
            np.random.seed(random_state)

        # Sample from multivariate t
        # Method: X = mu + Y * sqrt(df/S) where Y ~ N(0, Σ), S ~ chi2(df)
        normal_samples = np.random.multivariate_normal(
            np.zeros(self.dimension),
            self.correlation,
            size=size
        )

        chi2_samples = np.random.chisquare(self.df, size=size)

        t_samples = normal_samples * np.sqrt(self.df / chi2_samples)[:, np.newaxis]

        # Transform to uniform using t CDF
        u = stats.t.cdf(t_samples, df=self.df)

        return u

    def kendall_tau(self) -> float:
        """
        Calculate Kendall's tau (bivariate only).

        Returns:
            Kendall's tau
        """
        if self.dimension != 2:
            raise ValueError("Kendall's tau only defined for bivariate")

        rho = self.correlation[0, 1]
        return (2 / np.pi) * np.arcsin(rho)

    def __repr__(self) -> str:
        return f"StudentTCopula(dimension={self.dimension}, df={self.df})"


def fit_copula_to_data(data: np.ndarray,
                      copula_type: str = 'gaussian',
                      method: str = 'rank') -> Copula:
    """
    Fit copula to multivariate data.

    Args:
        data: Multivariate data (n x d)
        copula_type: Type of copula ('gaussian', 'clayton', 'gumbel', 't')
        method: Method for pseudo-observations ('rank' or 'empirical')

    Returns:
        Fitted copula object
    """
    n, d = data.shape

    # Transform to pseudo-observations (uniform margins)
    if method == 'rank':
        # Rank-based transformation
        u = np.zeros_like(data)
        for i in range(d):
            ranks = stats.rankdata(data[:, i])
            u[:, i] = ranks / (n + 1)
    else:
        # Empirical CDF
        u = np.zeros_like(data)
        for i in range(d):
            u[:, i] = stats.rankdata(data[:, i]) / n

    # Estimate copula parameters
    if copula_type == 'gaussian':
        # Estimate correlation from Gaussian quantiles
        z = stats.norm.ppf(u)
        corr = np.corrcoef(z.T)
        return GaussianCopula(corr)

    elif copula_type == 'clayton' and d == 2:
        # Estimate theta using Kendall's tau
        tau = stats.kendalltau(data[:, 0], data[:, 1])[0]
        theta = 2 * tau / (1 - tau)
        return ClaytonCopula(theta, dimension=2)

    elif copula_type == 'gumbel' and d == 2:
        # Estimate theta using Kendall's tau
        tau = stats.kendalltau(data[:, 0], data[:, 1])[0]
        theta = 1 / (1 - tau)
        return GumbelCopula(theta, dimension=2)

    elif copula_type == 't':
        # Estimate correlation
        # For df, use MLE (simplified: use fixed df=4)
        z = stats.t.ppf(u, df=4)
        corr = np.corrcoef(z.T)
        return StudentTCopula(corr, df=4)

    else:
        raise ValueError(f"Unsupported copula type: {copula_type}")
