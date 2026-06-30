"""Distribution modules for probability distributions."""

from .base import Distribution
from .continuous import (
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
)
from .discrete import (
    BinomialDistribution,
    PoissonDistribution,
    GeometricDistribution,
    NegativeBinomialDistribution,
    HypergeometricDistribution,
    DiscreteUniformDistribution,
)
from .multivariate import (
    MultivariateNormalDistribution,
    DirichletDistribution,
    MultivariateStudentT,
    WishartDistribution,
    plot_bivariate_normal,
    plot_dirichlet_simplex,
)
from .copulas import (
    GaussianCopula,
    ClaytonCopula,
    GumbelCopula,
    StudentTCopula,
    fit_copula_to_data,
)
from .mixtures import (
    MixtureDistribution,
    GaussianMixtureModel,
    BayesianGMM,
    select_optimal_components,
)

__all__ = [
    # Base
    "Distribution",
    # Continuous
    "NormalDistribution",
    "ExponentialDistribution",
    "UniformDistribution",
    "BetaDistribution",
    "GammaDistribution",
    "ChiSquareDistribution",
    "StudentTDistribution",
    "WeibullDistribution",
    "LognormalDistribution",
    "CauchyDistribution",
    # Discrete
    "BinomialDistribution",
    "PoissonDistribution",
    "GeometricDistribution",
    "NegativeBinomialDistribution",
    "HypergeometricDistribution",
    "DiscreteUniformDistribution",
    # Multivariate
    "MultivariateNormalDistribution",
    "DirichletDistribution",
    "MultivariateStudentT",
    "WishartDistribution",
    "plot_bivariate_normal",
    "plot_dirichlet_simplex",
    # Copulas
    "GaussianCopula",
    "ClaytonCopula",
    "GumbelCopula",
    "StudentTCopula",
    "fit_copula_to_data",
    # Mixtures
    "MixtureDistribution",
    "GaussianMixtureModel",
    "BayesianGMM",
    "select_optimal_components",
]
