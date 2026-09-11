"""Distribution modules for probability distributions."""

from .base import Distribution
from .continuous import (
    BetaDistribution,
    CauchyDistribution,
    ChiSquareDistribution,
    ExponentialDistribution,
    GammaDistribution,
    LognormalDistribution,
    NormalDistribution,
    StudentTDistribution,
    UniformDistribution,
    WeibullDistribution,
)
from .copulas import (
    ClaytonCopula,
    Copula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    fit_copula_to_data,
)
from .discrete import (
    BinomialDistribution,
    DiscreteUniformDistribution,
    GeometricDistribution,
    HypergeometricDistribution,
    NegativeBinomialDistribution,
    PoissonDistribution,
)
from .mixtures import (
    BayesianGMM,
    GaussianMixtureModel,
    MixtureDistribution,
    select_optimal_components,
)
from .multivariate import (
    DirichletDistribution,
    MultivariateDistribution,
    MultivariateNormalDistribution,
    MultivariateStudentT,
    WishartDistribution,
    plot_bivariate_normal,
    plot_dirichlet_simplex,
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
    "MultivariateDistribution",
    "MultivariateNormalDistribution",
    "DirichletDistribution",
    "MultivariateStudentT",
    "WishartDistribution",
    "plot_bivariate_normal",
    "plot_dirichlet_simplex",
    # Copulas
    "Copula",
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
