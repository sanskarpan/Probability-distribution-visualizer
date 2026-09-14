from importlib.metadata import version

from probviz.distributions import NormalDistribution
from probviz.fitting import DistributionFitter
from probviz.monte_carlo import MonteCarloSimulator

from probviz import __version__


def test_public_namespace_exposes_package_api() -> None:
    assert __version__ == version("probviz")
    assert NormalDistribution(mu=0, sigma=1).cdf(0) == 0.5
    assert DistributionFitter is not None
    assert MonteCarloSimulator is not None
