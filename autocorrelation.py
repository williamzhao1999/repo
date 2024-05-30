from __future__ import print_function, division
import matplotlib.pylab as plt
import numpy as np
from numpy.random import randn, choice, uniform, multivariate_normal
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import multivariate_normal as multivariate_norm
import json
import time
from scipy.stats import gamma
from utils import multiple_logpdfs, plot
import torch
from scipy.special import logsumexp
from scipy import stats
import os
import math
from pandas.plotting import autocorrelation_plot
import pandas as pd
f = open('./result_100000.json')
results = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/lambdas.json')
lambdas = np.array(json.load(f))
f.close()

N_parameters = lambdas.shape[0]

num_iterations = results.shape[0]
num_burn_iterations = math.floor((num_iterations*10000)/100000)

results = results[0:num_iterations,:]

burned_trace_mean = np.zeros(N_parameters)
var = 0

lambdas_results = np.zeros(N_parameters)

dir_path = "./images/autocorrelation"

if os.path.isdir(dir_path + "/" + str(num_iterations) + "/") == False:
    os.makedirs(dir_path + "/" + str(num_iterations) + "/") 

for t in range(N_parameters):
    trace = results[num_burn_iterations:num_iterations, t]
    lambdas_results[t] = np.mean(trace)
    burned_trace_mean[t] = (lambdas[t] - lambdas_results[t]) ** 2

    # Mean-center the series
    phi_mean_centered = trace - np.mean(trace)

    # Calculate the autocorrelation using np.correlate
    macf = np.correlate(phi_mean_centered, phi_mean_centered, mode='full')

    # Since np.correlate returns a result of length (2*N - 1), where N is the length of phiTrace
    # We take only the second half
    idx = int(macf.size / 2)
    macf = macf[idx:]

    # Normalize the autocorrelation function by the first element (variance)
    macf /= macf[0]

    # Create a grid for plotting
    grid = range(len(macf))

    # Plotting
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(grid, macf, color='#7570B3')
    plt.xlabel("Lag")
    plt.ylabel("ACF of lambda")
    plt.title("Autocorrelation Function (ACF) for the Entire Data Series")
    plt.tight_layout()
    plt.grid()
    plt.savefig(dir_path + "/" + str(num_iterations) + "/"+ f"zautocorrelation_{t}.png")
    plt.close()
    plt.show()
    '''
    autocorr_sum = 2 * np.sum(macf[1:])
    ESS = (num_iterations-num_burn_iterations) / (1 + autocorr_sum)

    print(f"ESS: {ESS}")

    best_iteration = 26120
    trace = results[2612:best_iteration, t]

    # Mean-center the series
    phi_mean_centered = trace - np.mean(trace)

    # Calculate the autocorrelation using np.correlate
    macf = np.correlate(phi_mean_centered, phi_mean_centered, mode='full')

    # Since np.correlate returns a result of length (2*N - 1), where N is the length of phiTrace
    # We take only the second half
    idx = int(macf.size / 2)
    macf = macf[idx:]

    # Normalize the autocorrelation function by the first element (variance)
    macf /= macf[0]
    autocorr_sum = 2 * np.sum(macf[1:])
    
    ESS2 = (best_iteration-2612) / (1 + autocorr_sum)

    print(f"ESS2: {ESS2}")