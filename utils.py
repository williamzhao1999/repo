import numpy as np

from pathlib import Path
from scipy.stats import uniform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math


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

def multiple_logpdfs(x, means, covs):
    """Compute multivariate normal log PDF over multiple sets of parameters.
    """
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets    = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs   = 1./vals
    
    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us         = vecs * np.sqrt(valsinvs)[:, None]
    devs       = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs      = np.einsum('ni,nij->nj', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas      = np.sum(np.square(devUs), axis=1)
    
    # Compute and broadcast scalar normalizers.
    dim        = len(vals[0])
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + mahas + logdets)

def multiple_logpdfs_gpu(x, means, covs, device, pi2):
    """Compute multivariate normal log PDF over multiple sets of parameters.
    """
    # NumPy broadcasts `eigh`.
    vals, vecs = torch.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets    = torch.sum(torch.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs   = 1./vals
    
    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us         = vecs * torch.sqrt(valsinvs)[:, None]
    devs       = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs      = torch.einsum('ni,nij->nj', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas      = torch.sum(torch.square(devUs), axis=1)
    
    # Compute and broadcast scalar normalizers.
    dim        = torch.tensor(vals[0].size(dim=0)).to(device)
    log2pi     = torch.log(pi2)
    return -torch.tensor(0.5).to(device) * (dim * log2pi + mahas + logdets)