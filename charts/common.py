import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.mstats import gmean
from contextlib import contextmanager
from matplotlib.ticker import MaxNLocator

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

@contextmanager
def figure(width, height, nrows, ncols, name=None, **kwargs):
    name = name if name is not None else \
        os.path.basename(sys.argv[0]).replace('.py', '')

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), **kwargs)
    yield fig, axs
    print(f'Saving {name}.pdf')
    plt.savefig(f'{name}.pdf')


def bars(
    ax,
    labels, # [N]
    data,   # [N]
    baseline=False,
    baseline_y=1,
    bar_width=0.75,
    ylim=None,
    locator_bins=10,
    labeloverflow=False,
    labelrot=90,
    color=colors[0],
    intylabels=True
):
    nbars = len(data)
    xs = np.arange(nbars)

    ax.bar(xs, data, bar_width, color=color, label=None)

    if baseline:
        ax.plot(
            [-bar_width, nbars - 1 + bar_width],
            [baseline_y, baseline_y],
            color='black',
            linestyle='--',
            linewidth=1.0)

    if labeloverflow:
        for i in range(len(data)):
            if data[i] > ylim:
                ax.text(
                    xs[i],
                    ylim * 0.92,
                    f'{int(data[i])}',
                    ha='center',
                    fontsize=6,
                    zorder=999)

    ax.set_xlim([-bar_width / 2, nbars - 1 + bar_width / 2])
    if ylim is not None: ax.set_ylim([0, ylim])
    if intylabels: ax.yaxis.set_major_locator(MaxNLocator(nbins=locator_bins, integer=True))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=labelrot)
    ax.tick_params(labelsize=8)

    return xs

def multibars(
    ax,
    grouplabels, # [ngrps]
    setlabels,   # [nsets]
    data,   # [nsets, ngrps]
    baseline=False,
    bar_width=0.75,
    pad_width=0.1,
    ylim=None,
    labeloverflow=False,
    labelrot=90,
    setcolors=None, # [nsets]
    intylabels=True,
    set_xticks=True,
    overflow_ypos_pct=0.92,
    locator_bins=10
):
    nsets = len(setlabels)
    ngrps = len(grouplabels)

    setcolors = setcolors if setcolors is not None else colors[:nsets]

    xs = np.arange(ngrps) * (nsets * bar_width + pad_width) + bar_width / 2
    label_xs = (xs - bar_width / 2) + (nsets * bar_width) / 2

    for i in range(nsets):
        xs_i = xs + i * bar_width
        ax.bar(
            xs_i,
            data[i, :],
            bar_width,
            color=setcolors[i],
            label=setlabels[i])

        if labeloverflow:
            for j in range(ngrps):
                if int(data[i, j]) > ylim:
                    ax.text(
                        xs_i[j],
                        ylim * overflow_ypos_pct,
                        f'{int(data[i, j])}',
                        ha='center',
                        fontsize=6,
                        zorder=999)

    if baseline:
        ax.plot(
            [0, ngrps * (nsets * bar_width + pad_width) - pad_width],
            [1, 1],
            color='black',
            linestyle='--',
            linewidth=1.0)

    ax.set_xlim([0, ngrps * (nsets * bar_width + pad_width) - pad_width])
    if intylabels: ax.yaxis.set_major_locator(MaxNLocator(nbins=locator_bins, integer=True))
    if ylim is not None: ax.set_ylim([0, ylim])

    if set_xticks:
        ax.set_xticks(label_xs)
        ax.set_xticklabels(grouplabels, rotation=labelrot)

    ax.tick_params(labelsize=8)

def stacked_multibars(
    ax,
    grouplabels, # [ngrps]
    setlabels,   # [nsets]
    data,   # [nsets, ngrps]
    baseline=False,
    bar_width=0.75,
    pad_width=0.1,
    ylim=None,
    labeloverflow=False,
    labelrot=90,
    setcolors=None, # [nsets]
    intylabels=True,
    set_xticks=True,
    overflow_ypos_pct=0.92,
    locator_bins=10
):
    nsets = len(setlabels)
    ngrps = len(grouplabels)

    setcolors = setcolors if setcolors is not None else colors[:nsets]

    xs = np.arange(ngrps) * (1 * bar_width + pad_width) + bar_width / 2
    label_xs = (xs - bar_width / 2) + (1 * bar_width) / 2

    for i in range(nsets):
        ax.bar(
            xs,
            data[i, :],
            bar_width,
            color=setcolors[i],
            label=setlabels[i],
            zorder=-i)

        if labeloverflow:
            for j in range(ngrps):
                if int(data[i, j]) > ylim:
                    ax.text(
                        xs[j],
                        ylim * overflow_ypos_pct,
                        f'{int(data[i, j])}',
                        ha='center',
                        fontsize=6,
                        zorder=999)

    if baseline:
        ax.plot(
            [0, ngrps * (1 * bar_width + pad_width) - pad_width],
            [1, 1],
            color='black',
            linestyle='--',
            linewidth=1.0)

    ax.set_xlim([0, ngrps * (1 * bar_width + pad_width) - pad_width])
    if intylabels: ax.yaxis.set_major_locator(MaxNLocator(nbins=locator_bins, integer=True))
    if ylim is not None: ax.set_ylim([0, ylim])

    if set_xticks:
        ax.set_xticks(label_xs)
        ax.set_xticklabels(grouplabels, rotation=labelrot)

    ax.tick_params(labelsize=8)