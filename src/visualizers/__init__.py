"""Visualization helpers for probability distributions.

This package provides the public visualization API. The heavy lifting lives in
:mod:`src.utils.plotting`; the functions here add distribution-aware wrappers so
both ``src.visualizers`` and ``src.utils.plotting`` import paths work.
"""

from typing import Any, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import (
    plot_correlation_heatmap,
    plot_distribution_comparison,
    plot_histogram_with_fit,
    plot_probability_bands,
    plot_qq,
)


def _infer_grid(distribution: Any, x: Any, num_points: int) -> np.ndarray:
    """Build an evaluation grid, preferring the distribution's own support."""
    if x is not None:
        return np.asarray(x)
    lower, upper = -5.0, 5.0
    for accessor in ("get_support", "support"):
        try:
            lo, hi = getattr(distribution, accessor)()
            lower, upper = float(lo), float(hi)
            break
        except Exception:
            continue
    if not np.isfinite(lower):
        lower = -5.0
    if not np.isfinite(upper):
        upper = 5.0
    if lower == upper:
        lower, upper = lower - 1.0, upper + 1.0
    return np.linspace(lower, upper, num_points)


__all__ = [
    "plot_distribution_comparison",
    "plot_qq",
    "plot_histogram_with_fit",
    "plot_correlation_heatmap",
    "plot_probability_bands",
    "plot_pdf",
    "plot_cdf",
    "plot_comparison",
]


def plot_pdf(
    distribution: Any,
    x: Optional[np.ndarray] = None,
    num_points: int = 500,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the PDF/PMF of a distribution.

    Args:
        distribution: Any object exposing ``pdf(x)``.
        x: Evaluation points. If None, inferred from the distribution support.
        num_points: Points to use when inferring the grid.
        ax: Optional matplotlib axes to draw on.

    Returns:
        Tuple of (figure, axes).
    """
    x = _infer_grid(distribution, x, num_points)
    y = np.asarray(distribution.pdf(x))
    if ax is None:
        fig, ax_out = plt.subplots()
    else:
        ax_out = ax
        # An Axes handed in by the caller always belongs to a Figure;
        # the stub type is wider (Figure | SubFigure | None).
        fig = cast(plt.Figure, ax_out.figure)
    ax_out.plot(np.asarray(x), y)
    ax_out.set_xlabel("x")
    ax_out.set_ylabel("Density / Mass")
    ax_out.set_title(f"{distribution!r} - PDF/PMF")
    ax_out.grid(True, alpha=0.3)
    return fig, ax_out


def plot_cdf(
    distribution: Any,
    x: Optional[np.ndarray] = None,
    num_points: int = 500,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the CDF of a distribution (see :func:`plot_pdf` for args)."""
    x = _infer_grid(distribution, x, num_points)
    y = np.asarray(distribution.cdf(x))
    if ax is None:
        fig, ax_out = plt.subplots()
    else:
        ax_out = ax
        # An Axes handed in by the caller always belongs to a Figure;
        # the stub type is wider (Figure | SubFigure | None).
        fig = cast(plt.Figure, ax_out.figure)
    ax_out.plot(np.asarray(x), y)
    ax_out.set_xlabel("x")
    ax_out.set_ylabel("Cumulative probability")
    ax_out.set_title(f"{distribution!r} - CDF")
    ax_out.grid(True, alpha=0.3)
    return fig, ax_out


def plot_comparison(
    distributions: List, x: Optional[np.ndarray] = None, num_points: int = 500
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay the PDFs/PMFs of several distributions on shared axes."""
    fig, ax = plt.subplots()
    for dist in distributions:
        plot_pdf(dist, x=x, num_points=num_points, ax=ax)
    ax.legend([f"{d!r}" for d in distributions])
    ax.set_title("Distribution comparison")
    return fig, ax
