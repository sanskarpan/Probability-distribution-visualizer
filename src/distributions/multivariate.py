"""Multivariate probability distributions."""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MultivariateDistribution:
    """Base class for multivariate distributions."""

    def __init__(self, name: str, dimension: int):
        """
        Initialize multivariate distribution.

        Args:
            name: Name of the distribution
            dimension: Number of dimensions
        """
        self.name = name
        self.dimension = dimension
        self._dist = None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Calculate probability density function."""
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        raise NotImplementedError

    def mean(self) -> np.ndarray:
        """Calculate mean vector."""
        raise NotImplementedError

    def cov(self) -> np.ndarray:
        """Calculate covariance matrix."""
        raise NotImplementedError


class MultivariateNormalDistribution(MultivariateDistribution):
    """Multivariate Normal (Gaussian) distribution."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Initialize Multivariate Normal distribution.

        Args:
            mean: Mean vector (shape: d)
            cov: Covariance matrix (shape: d x d)
        """
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        if mean.ndim != 1:
            raise ValueError("mean must be 1-dimensional")

        if cov.ndim != 2:
            raise ValueError("cov must be 2-dimensional")

        if cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be square")

        if len(mean) != cov.shape[0]:
            raise ValueError("mean and cov dimensions must match")

        # Check if covariance matrix is positive definite
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            raise ValueError("cov must be positive definite")

        super().__init__("Multivariate Normal", len(mean))
        self.mean_vec = mean
        self.cov_mat = cov
        self._dist = stats.multivariate_normal(mean=mean, cov=cov)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate probability density function.

        Args:
            x: Points to evaluate (shape: n x d or d)

        Returns:
            PDF values
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.pdf(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Calculate log probability density function."""
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.logpdf(x)

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples.

        Args:
            size: Number of samples
            random_state: Random seed

        Returns:
            Samples (shape: size x d)
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.rvs(size=size, random_state=random_state)

    def mean(self) -> np.ndarray:
        """Calculate mean vector."""
        return self.mean_vec

    def cov(self) -> np.ndarray:
        """Calculate covariance matrix."""
        return self.cov_mat

    def marginal(self, indices: list) -> 'MultivariateNormalDistribution':
        """
        Get marginal distribution for selected variables.

        Args:
            indices: List of variable indices to keep

        Returns:
            Marginal distribution
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        indices = np.array(indices)
        marginal_mean = self.mean_vec[indices]
        marginal_cov = self.cov_mat[np.ix_(indices, indices)]
        return MultivariateNormalDistribution(marginal_mean, marginal_cov)

    def conditional(self, indices: list, values: np.ndarray) -> 'MultivariateNormalDistribution':
        """
        Get conditional distribution.

        Args:
            indices: Indices of variables to condition on
            values: Values of conditioned variables

        Returns:
            Conditional distribution
        """
        indices_arr = np.array(indices)
        free_indices = np.array([i for i in range(self.dimension) if i not in indices_arr])

        # Partition mean and covariance
        mu1 = self.mean_vec[free_indices]
        mu2 = self.mean_vec[indices]
        sigma11 = self.cov_mat[np.ix_(free_indices, free_indices)]
        sigma12 = self.cov_mat[np.ix_(free_indices, indices)]
        sigma22 = self.cov_mat[np.ix_(indices, indices)]

        # Conditional parameters
        sigma22_inv = np.linalg.inv(sigma22)
        cond_mean = mu1 + sigma12 @ sigma22_inv @ (values - mu2)
        cond_cov = sigma11 - sigma12 @ sigma22_inv @ sigma12.T

        return MultivariateNormalDistribution(cond_mean, cond_cov)

    def mahalanobis(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate Mahalanobis distance.

        Args:
            x: Points (shape: n x d or d)

        Returns:
            Mahalanobis distances
        """
        x = np.atleast_2d(x)
        diff = x - self.mean_vec
        cov_inv = np.linalg.inv(self.cov_mat)
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    def __repr__(self) -> str:
        return f"MultivariateNormal(dimension={self.dimension})"


class DirichletDistribution(MultivariateDistribution):
    """Dirichlet distribution."""

    def __init__(self, alpha: np.ndarray):
        """
        Initialize Dirichlet distribution.

        Args:
            alpha: Concentration parameters (must be positive)
        """
        alpha = np.asarray(alpha)

        if alpha.ndim != 1:
            raise ValueError("alpha must be 1-dimensional")

        if np.any(alpha <= 0):
            raise ValueError("alpha must be positive")

        super().__init__("Dirichlet", len(alpha))
        self.alpha = alpha
        self._dist = stats.dirichlet(alpha)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate probability density function.

        Args:
            x: Points on simplex (shape: n x d or d), must sum to 1

        Returns:
            PDF values
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.pdf(x.T if x.ndim == 2 else x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Calculate log probability density function."""
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.logpdf(x.T if x.ndim == 2 else x)

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples.

        Args:
            size: Number of samples
            random_state: Random seed

        Returns:
            Samples on simplex (shape: size x d)
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.rvs(size=size, random_state=random_state)

    def mean(self) -> np.ndarray:
        """Calculate mean vector."""
        return self.alpha / np.sum(self.alpha)

    def var(self) -> np.ndarray:
        """Calculate variance for each component."""
        alpha0 = np.sum(self.alpha)
        return (self.alpha * (alpha0 - self.alpha)) / (alpha0**2 * (alpha0 + 1))

    def cov(self) -> np.ndarray:
        """Calculate covariance matrix."""
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.cov()

    def mode(self) -> np.ndarray:
        """
        Calculate mode.

        Returns:
            Mode vector (only valid if all alpha > 1)
        """
        if np.any(self.alpha <= 1):
            raise ValueError("Mode only defined when all alpha > 1")

        return (self.alpha - 1) / (np.sum(self.alpha) - self.dimension)

    def entropy(self) -> float:
        """Calculate differential entropy."""
        if self._dist is None:
            raise ValueError("Distribution not initialized")
        return self._dist.entropy()

    def __repr__(self) -> str:
        return f"Dirichlet(dimension={self.dimension}, alpha={self.alpha})"


class MultivariateStudentT(MultivariateDistribution):
    """Multivariate Student's t-distribution."""

    def __init__(self, df: float, loc: np.ndarray, shape: np.ndarray):
        """
        Initialize Multivariate Student-t distribution.

        Args:
            df: Degrees of freedom
            loc: Location vector
            shape: Shape matrix (similar to covariance)
        """
        loc = np.asarray(loc)
        shape = np.asarray(shape)

        if df <= 0:
            raise ValueError("df must be positive")

        if loc.ndim != 1:
            raise ValueError("loc must be 1-dimensional")

        if shape.ndim != 2 or shape.shape[0] != shape.shape[1]:
            raise ValueError("shape must be square matrix")

        super().__init__("Multivariate Student-t", len(loc))
        self.df = df
        self.loc_vec = loc
        self.shape_mat = shape

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Calculate probability density function."""
        x = np.atleast_2d(x)
        d = self.dimension
        df = self.df

        diff = x - self.loc_vec
        shape_inv = np.linalg.inv(self.shape_mat)
        shape_det = np.linalg.det(self.shape_mat)

        mahalanobis_sq = np.sum(diff @ shape_inv * diff, axis=1)

        # Compute normalizing constant
        from scipy.special import gamma
        numer = gamma((df + d) / 2)
        denom = gamma(df / 2) * ((df * np.pi) ** (d / 2)) * np.sqrt(shape_det)
        normalizing = numer / denom

        # Compute PDF
        pdf_vals = normalizing * (1 + mahalanobis_sq / df) ** (-(df + d) / 2)

        return pdf_vals

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        if random_state is not None:
            np.random.seed(random_state)

        # Generate using property: t_d = loc + sqrt(d/chi^2_d) * N(0, shape)
        chi2_samples = np.random.chisquare(self.df, size=size)
        normal_samples = np.random.multivariate_normal(
            np.zeros(self.dimension),
            self.shape_mat,
            size=size
        )

        samples = self.loc_vec + normal_samples * np.sqrt(self.df / chi2_samples)[:, np.newaxis]
        return samples

    def mean(self) -> np.ndarray:
        """Calculate mean vector (only for df > 1)."""
        if self.df <= 1:
            raise ValueError("Mean only defined for df > 1")
        return self.loc_vec

    def cov(self) -> np.ndarray:
        """Calculate covariance matrix (only for df > 2)."""
        if self.df <= 2:
            raise ValueError("Covariance only defined for df > 2")
        return self.shape_mat * (self.df / (self.df - 2))

    def __repr__(self) -> str:
        return f"MultivariateStudentT(dimension={self.dimension}, df={self.df})"


class WishartDistribution:
    """Wishart distribution (distribution over positive definite matrices)."""

    def __init__(self, df: int, scale: np.ndarray):
        """
        Initialize Wishart distribution.

        Args:
            df: Degrees of freedom (must be >= dimension)
            scale: Scale matrix (positive definite)
        """
        scale = np.asarray(scale)

        if scale.ndim != 2 or scale.shape[0] != scale.shape[1]:
            raise ValueError("scale must be square matrix")

        dimension = scale.shape[0]

        if df < dimension:
            raise ValueError("df must be >= dimension")

        self.name = "Wishart"
        self.dimension = dimension
        self.df = df
        self.scale_mat = scale
        self._dist = stats.wishart(df=df, scale=scale)

    def pdf(self, x: np.ndarray) -> float:
        """Calculate probability density function."""
        return self._dist.pdf(x)

    def logpdf(self, x: np.ndarray) -> float:
        """Calculate log probability density function."""
        return self._dist.logpdf(x)

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate random positive definite matrices."""
        return self._dist.rvs(size=size, random_state=random_state)

    def mean(self) -> np.ndarray:
        """Calculate mean matrix."""
        return self._dist.mean()

    def mode(self) -> np.ndarray:
        """Calculate mode matrix."""
        if self.df >= self.dimension + 1:
            return (self.df - self.dimension - 1) * self.scale_mat
        raise ValueError("Mode only defined for df >= dimension + 1")

    def __repr__(self) -> str:
        return f"Wishart(dimension={self.dimension}, df={self.df})"


def plot_bivariate_normal(dist: MultivariateNormalDistribution,
                          num_points: int = 100,
                          num_contours: int = 10) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot bivariate normal distribution.

    Args:
        dist: Multivariate normal distribution (dimension must be 2)
        num_points: Number of grid points
        num_contours: Number of contour levels

    Returns:
        Figure and axes objects
    """
    if dist.dimension != 2:
        raise ValueError("Can only plot bivariate distributions")

    # Create grid
    mean = dist.mean()
    cov = dist.cov()

    # Determine plot limits based on covariance
    std1 = np.sqrt(cov[0, 0])
    std2 = np.sqrt(cov[1, 1])

    x1 = np.linspace(mean[0] - 3*std1, mean[0] + 3*std1, num_points)
    x2 = np.linspace(mean[1] - 3*std2, mean[1] + 3*std2, num_points)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate PDF
    pos = np.dstack((X1, X2))
    Z = dist.pdf(pos)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot
    contour = ax1.contourf(X1, X2, Z, levels=num_contours, cmap='viridis')
    ax1.contour(X1, X2, Z, levels=num_contours, colors='white', alpha=0.3, linewidths=0.5)
    fig.colorbar(contour, ax=ax1, label='Probability Density')
    ax1.plot(mean[0], mean[1], 'r*', markersize=15, label='Mean')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_title('Bivariate Normal Distribution - Contour Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('Probability Density')
    ax2.set_title('Bivariate Normal Distribution - 3D Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_dirichlet_simplex(dist: DirichletDistribution,
                           num_samples: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Dirichlet distribution samples on simplex (for dimension 3).

    Args:
        dist: Dirichlet distribution (dimension must be 3)
        num_samples: Number of samples to generate

    Returns:
        Figure and axes objects
    """
    if dist.dimension != 3:
        raise ValueError("Can only plot 3-dimensional Dirichlet on simplex")

    # Generate samples
    samples = dist.rvs(size=num_samples, random_state=42)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                        c=samples[:, 0], cmap='viridis', alpha=0.6, s=20)

    # Plot simplex edges
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([vertices[i, 0], vertices[j, 0]],
                   [vertices[i, 1], vertices[j, 1]],
                   [vertices[i, 2], vertices[j, 2]],
                   'k-', linewidth=2, alpha=0.5)

    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_zlabel('X₃')
    ax.set_title(f'Dirichlet Distribution Samples\nα = {dist.alpha}')
    fig.colorbar(scatter, ax=ax, label='X₁ value', shrink=0.5)

    return fig, ax
