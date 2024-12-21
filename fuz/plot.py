"""Plotting functions."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np
from mpltern.datasets import get_dirichlet_pdfs
from scipy.stats._distn_infrastructure import rv_discrete_frozen

import fuz.dists as fd


def terngrid(nmax: int, norm_range: None | tuple = None) -> np.ndarray:
    """Generate a ternary grid.

    Parameters
    ----------
    nmax
        The size of one axis of the grid.
    norm_range
        The range to normalize the grid to. If None, the grid points are in the range (0, nmax).

    Returns
    -------
    np.ndarray
        The generated ternary grid.
    """
    grid = np.array(
        [(i, j, nmax - i - j) for i in range(nmax + 1) for j in range(nmax - i + 1)]
    )
    if norm_range is not None:
        grid = np.interp(grid, (0, nmax), norm_range)
    return grid


def plot_multinomial(
    dist: rv_discrete_frozen, title: str | None = None, ax_labels: tuple | None = None
):
    grid = terngrid(dist.n)
    pmf = dist.pmf(grid)
    t, l, r = grid[:, 0], grid[:, 1], grid[:, 2]
    fig = plt.figure()
    fig.subplots_adjust(top=0.85, bottom=0.15)
    ax = plt.subplot(projection='ternary', ternary_sum=dist.n)
    shading = ax.tripcolor(t, l, r, pmf, cmap='turbo', shading='flat', rasterized=True)
    contour = ax.tricontour(t, l, r, pmf, colors='k', alpha=0.5, levels=3, linewidths=0.5)
    ax.clabel(contour, fontsize=8)
    t, l, r = dist.mean()
    ax.scatter(
        t, l, r, color='skyblue', label=f'$\\mu$: {t:0.2g}, {l:0.2g}, {r:0.2g}', marker='x'
    )
    if ax_labels is not None:
        ax.set_tlabel(f'{ax_labels[0]}')
        ax.set_llabel(f'{ax_labels[1]}')
        ax.set_rlabel(f'{ax_labels[2]}')
    else:
        ax.set_tlabel('1')
        ax.set_llabel('2')
        ax.set_rlabel('3')
    ax.taxis.set_tick_params(labelsize=8)
    ax.laxis.set_tick_params(labelsize=8)
    ax.raxis.set_tick_params(labelsize=8)
    ax.grid()
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.1)
    ax.set_title(
        rf'$\mathbf{{p}} = {dist.p[0]:0.2g}, {dist.p[1]:0.2g}, {dist.p[2]:0.2g}$', fontsize=9
    )
    if title:
        fig.subplots_adjust(top=0.75, bottom=0.15)
        fig.suptitle(title, fontsize=14)
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(shading, cax=cax)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label('probability', rotation=270, va='baseline')
    return fig, ax


def plot_multinomial_3d(dist: rv_discrete_frozen):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    grid = terngrid(dist.n)
    pmf = dist.pmf(grid)
    xs, ys, zs = grid[:, 0], grid[:, 1], grid[:, 2]
    ax.scatter(xs, ys, zs, s=pmf * 1000, c=pmf, cmap='turbo')
    return fig, ax


def plot_scored_pdf(dist: fd.Scored, title: str | None = None):
    fig = plt.figure()
    fig.subplots_adjust(top=0.75, bottom=0.15)
    ax = plt.subplot(projection='ternary')

    t, l, r, v = get_dirichlet_pdfs(n=37, alpha=dist.alpha)
    shading = ax.tripcolor(t, l, r, v, cmap='turbo', shading='gouraud', rasterized=True)

    contour = ax.tricontour(t, l, r, v, colors='k', alpha=0.3, levels=3, linewidths=0.5)
    ax.clabel(contour, fontsize=8)
    t, l, r = dist.mode
    ax.scatter(t, l, r, color='aquamarine', label=f'mode: {dist.mo:0.2f}', marker='.')
    t, l, r = dist.mean
    ax.scatter(t, l, r, color='skyblue', label=f'$\\mu$: {dist.mu:0.2f}', marker='x')
    ax.set_tlabel(f'{dist.scores[0]}')
    ax.set_llabel(f'{dist.scores[1]}')
    ax.set_rlabel(f'{dist.scores[2]}')
    ax.taxis.set_tick_params(labelsize=8)
    ax.laxis.set_tick_params(labelsize=8)
    ax.raxis.set_tick_params(labelsize=8)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.1)
    ax.set_title('$\\mathbf{\\alpha}$ = ' + str(dist.alpha), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=14)
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(shading, cax=cax)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label('probability density', rotation=270, va='baseline')
    return fig, ax
