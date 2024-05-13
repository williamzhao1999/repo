import numpy as np

from pathlib import Path
from scipy.stats import uniform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def update_diag_func(data, label, color):
    for val in data.quantile([.25, .5, .75]):
        plt.axvline(val, ls=':', color=color)
    plt.title(data.name, color=color)

def plot_graph(samples, weights=None, param_names=None,
                  save=False, show=True, xlim=None, ylim=None,
                  filename='pairwise.png'):  # pragma no cover
    #TODO true params on plot #true_params=None,
    '''
    Plots pairwise distributions of all parameter combos. Color codes each
    by weight if provided.
    '''
    if param_names is None:
        param_names = [f'p{i}' for i in range(samples.shape[1])]

    if weights is None:
        weights = np.ones((samples.shape[0], 1))

    columns = param_names + ['weights_']
    samples = np.hstack((samples, weights))

    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])

    df = pd.DataFrame(samples, columns=columns)

    ax = sns.pairplot(df, diag_kind='kde', corner=True,
                      hue='weights_', palette='viridis',
                      diag_kws={'weights': weights.flatten(), 'hue': None})
    ax.map_diag(update_diag_func)
    ax.set(xlim=xlim)
    ax.set(ylim=ylim)



    ax.legend.remove()
    cbar_ax = ax.fig.add_axes([0.65, 0.3, 0.02, 0.4])  # Adjust these values to change the position and size of the colorbar
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel('Particle Weight')

    # compute means TODO plot means
    if weights is None:
        means = np.mean(samples, axis=0)
    else:
        means = np.sum(samples * weights, axis=0)

    if save:
        plt.savefig(filename)
    if show:
        plt.show()

    return ax