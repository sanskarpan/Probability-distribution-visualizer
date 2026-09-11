"""Generate animated GIFs for README/docs.

Reproducible media pipeline: every GIF under ``docs/assets/`` is produced by
this script so visuals can be regenerated after library changes.

Usage:
    python examples/generate_media.py [--out docs/assets]

Requires: matplotlib + pillow (PillowWriter).
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distributions import BetaDistribution, MixtureDistribution, NormalDistribution


def _save(anim: FuncAnimation, path: str, fps: int = 3) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    anim.save(path, writer="pillow", fps=fps)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)")


def normal_sigma_morph(out: str) -> None:
    """Normal(0, sigma) PDF as sigma sweeps 0.5 -> 2.5."""
    x = np.linspace(-6, 6, 300)
    sigmas = np.linspace(0.5, 2.5, 15)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def frame(i: int):
        s = float(sigmas[i])
        line.set_data(x, NormalDistribution(mu=0, sigma=s).pdf(x))
        title.set_text(f"Normal(0, sigma={s:.2f})")
        return line, title

    _save(FuncAnimation(fig, frame, frames=len(sigmas), interval=350, blit=False), out)
    plt.close(fig)


def beta_shape_morph(out: str) -> None:
    """Beta(a, b) PDF morphing from uniform to skewed to symmetric."""
    x = np.linspace(0, 1, 300)
    params = [(1, 1), (2, 2), (5, 2), (2, 5), (0.7, 0.7), (5, 5), (8, 3), (3, 8)] * 2
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    (line,) = ax.plot([], [], lw=2, color="darkorange")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def frame(i: int):
        a, b = params[i]
        line.set_data(x, BetaDistribution(alpha=a, beta=b).pdf(x))
        title.set_text(f"Beta(a={a}, b={b})")
        return line, title

    _save(FuncAnimation(fig, frame, frames=len(params), interval=450, blit=False), out)
    plt.close(fig)


def clt_convergence(out: str) -> None:
    """CLT: distribution of Uniform(0,1) sample means for growing n."""
    from src.distributions import UniformDistribution

    uni = UniformDistribution(a=0, b=1)
    rng = np.random.default_rng(7)
    draws = uni.rvs(size=60_000, random_state=7)
    sample_sizes = [1, 2, 5, 10, 20, 30, 50, 80]
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def frame(i: int):
        ax.clear()
        n = sample_sizes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 9)
        ax.set_xlabel("Sample mean")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        means = draws[: (len(draws) // n) * n].reshape(-1, n).mean(axis=1)
        ax.hist(means, bins=40, density=True, alpha=0.65, color="steelblue", edgecolor="white")
        mu, sigma = 0.5, np.sqrt(1 / 12 / n)
        xs = np.linspace(0, 1, 200)
        ax.plot(xs, NormalDistribution(mu=mu, sigma=sigma).pdf(xs), "r-", lw=2, label="Normal fit")
        ax.set_title(f"CLT: Uniform means, n={n}")
        ax.legend(fontsize=8)
        return ax

    _save(FuncAnimation(fig, frame, frames=len(sample_sizes), interval=600, blit=False), out)
    plt.close(fig)
    _ = rng  # silence unused (rvs uses its own seeding path)


def copula_dependence(out: str) -> None:
    """Gaussian copula scatter as correlation sweeps 0 -> 0.9."""
    from src.distributions import GaussianCopula

    rhos = np.linspace(0.0, 0.9, 12)
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=80)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("U1")
    ax.set_ylabel("U2")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")
    scat = ax.scatter([], [], s=6, alpha=0.5)

    def frame(i: int):
        rho = float(rhos[i])
        u = GaussianCopula(np.array([[1.0, rho], [rho, 1.0]])).rvs(size=400, random_state=11)
        scat.set_offsets(u)
        title.set_text(f"Gaussian copula, rho={rho:.2f}")
        return scat, title

    _save(FuncAnimation(fig, frame, frames=len(rhos), interval=400, blit=False), out)
    plt.close(fig)


def mixture_separation(out: str) -> None:
    """Two-component Gaussian mixture PDF as the components separate."""
    x = np.linspace(-4, 10, 400)
    mus = np.linspace(0.5, 5.0, 14)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    (line,) = ax.plot([], [], lw=2, color="seagreen")
    ax.set_xlim(-4, 10)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def frame(i: int):
        mu = float(mus[i])
        mix = MixtureDistribution(
            [NormalDistribution(mu=0, sigma=1), NormalDistribution(mu=mu, sigma=1)],
            [0.5, 0.5],
        )
        line.set_data(x, mix.pdf(x))
        title.set_text(f"0.5 N(0,1) + 0.5 N({mu:.1f},1)")
        return line, title

    _save(FuncAnimation(fig, frame, frames=len(mus), interval=350, blit=False), out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate README/docs GIFs")
    parser.add_argument("--out", default="docs/assets", help="Output directory")
    args = parser.parse_args()

    normal_sigma_morph(os.path.join(args.out, "normal_sigma_morph.gif"))
    beta_shape_morph(os.path.join(args.out, "beta_shape_morph.gif"))
    clt_convergence(os.path.join(args.out, "clt_convergence.gif"))
    copula_dependence(os.path.join(args.out, "copula_dependence.gif"))
    mixture_separation(os.path.join(args.out, "mixture_separation.gif"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
