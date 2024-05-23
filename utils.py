import numpy as np

from pathlib import Path
from scipy.stats import uniform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import json
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

def RMSE(true, pred):
    return np.sqrt(np.mean((true - pred)**2))

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

def plot(trace, noBins, grid, true_value, dir_path, name, burned = True, parameter_name = "lambda"):


    # Plot the parameter posterior estimate (solid black line = posterior mean)
    plt.subplot(2, 1, 1)
    plt.hist(trace, noBins, density=True, facecolor='#7570B3')
    plt.xlabel(parameter_name)
    plt.ylabel("posterior density estimate")
    plt.axvline(np.mean(trace), color='k')
    plt.axvline(true_value, color='g')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(2, 1, 2)
    plt.plot(grid, trace, color='#7570B3')
    plt.xlabel("iteration")
    plt.ylabel(parameter_name)
    plt.axhline(np.mean(trace), color='k')
    plt.axhline(true_value, color='g')

    plt.savefig(f"{dir_path}/{name}.png")



    plt.close()

def generateData(noObservations, initialState):
    state = np.zeros((noObservations + 1, x_length))
    observation = np.zeros((noObservations, y_length))
    state[0] = initialState
    
    for t in range(1, noObservations):
        state[t] = np.matmul(A[t - 1], state[t - 1].T) + np.matmul(B[t - 1], u[t - 1])
        observation[t] = np.matmul(H[t], state[t])

    return(state, observation)

class EarlyStopping:
    def __init__(self, stop_after_iterations):
        self.phase1 = True # Reach max value variance
        self.phase1_count_to_pass_phase2 = 0
        self.phase2 = False # Reach min value after the max value of variance
        self.phase2_count_to_pass_phase3 = 0
        self.phase3 = False # Start to increase so stop
        self.phase3_count = 0
        self.variance = 0
        self.stop_after_iterations = 1000 #stop after 1000 hundreds iterations incease to the least value stored
    
    def verify(self, new_variance): # True stop, False no stop
        if self.phase1 == True:
            if self.variance == 0:
                self.variance = new_variance
            else:
                if self.variance > new_variance:
                    phase1_count_to_pass_phase2 = phase1_count_to_pass_phase2 + 1
                else:
                    self.variance = new_variance
                
                if phase1_count_to_pass_phase2 >= self.stop_after_iterations:
                    self.phase2 = True
                    self.phase1 = False
            
        elif self.phase2 == True:
            if self.variance > new_variance:
                self.variance = new_variance
            else:
                phase2_count_to_pass_phase3 = phase2_count_to_pass_phase3 + 1
                if phase2_count_to_pass_phase3 >= self.stop_after_iterations:
                    self.phase2 = False
                    self.phase3 = True
            
        elif self.phase3 == True:
            if self.variance <= new_variance:
                self.phase3_count = self.phase3_count + 1

                if self.phase3_count >= self.stop_after_iterations:
                    return True
        
        return False
        