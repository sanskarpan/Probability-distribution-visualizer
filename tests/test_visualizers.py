"""Tests for the src.visualizers facade."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from src.distributions import NormalDistribution
from src.visualizers import plot_cdf, plot_comparison, plot_pdf


def test_plot_pdf_returns_figure():
    dist = NormalDistribution(mu=0, sigma=1)
    fig, ax = plot_pdf(dist)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cdf_monotone():
    dist = NormalDistribution(mu=0, sigma=1)
    fig, ax = plot_cdf(dist)
    lines = ax.get_lines()
    assert len(lines) == 1
    y = lines[0].get_ydata()
    assert np.all(np.diff(y) >= 0)
    plt.close(fig)


def test_plot_comparison_overlays():
    from src.distributions import ExponentialDistribution

    dists = [NormalDistribution(mu=0, sigma=1), ExponentialDistribution(lambda_param=1.0)]
    # Exponential support starts at 0; pass an explicit shared grid.
    fig, ax = plot_comparison(dists, x=np.linspace(-3, 5, 100))
    assert len(ax.get_lines()) == len(dists)
    plt.close(fig)


def test_plot_pdf_with_explicit_ax():
    dist = NormalDistribution(mu=0, sigma=1)
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_pdf(dist, ax=ax)
    assert out_ax is ax
    assert out_fig is fig
    plt.close(fig)
