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
from utils import multiple_logpdfs, RMSE, plot
import torch
from scipy.special import logsumexp
from scipy import stats
import os
import math

f = open('./result_10000.json')
results = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/lambdas.json')
lambdas = np.array(json.load(f))
f.close()

N_parameters = lambdas.shape[0]

noBurnInIterations = 1000
noIterations = 10000

burned_trace_mean = np.zeros(N_parameters)
standard_deviation = 0

dir_path = "./images"

for t in range(N_parameters):
    trace = results[noBurnInIterations:noIterations, t]
    burned_trace_mean[t] = np.sqrt(np.mean( (lambdas[t] - trace) ** 2))
    standard_deviation += np.var(trace)

    noBins = int(np.floor(np.sqrt(noIterations - noBurnInIterations)))
    grid = np.arange(noBurnInIterations, noIterations, 1)
    plot(trace, noBins, grid, lambdas[t], dir_path, f"lambda_{t}")

    trace_noburned = results[:, t]
    noBins2 = int(np.floor(np.sqrt(noIterations)))
    grid2 = np.arange(0, noIterations, 1)
    plot(trace_noburned, noBins2, grid2, lambdas[t], dir_path, f"lambda_{t}_noburned")

print(f"RMSE: {np.sum(burned_trace_mean)}")
print(f"Std: {np.sqrt(standard_deviation)}")

RMSE = np.zeros(noIterations)
for i in range(noIterations):
    burned_trace_mean = np.zeros(N_parameters)
    for t in range(N_parameters):
        no_burn_iterations = math.floor((i*noBurnInIterations)/noIterations)
        trace = results[no_burn_iterations:i, t]
        burned_trace_mean[t] = np.sqrt(np.mean( (lambdas[t] - trace) ** 2))
    RMSE[i] = np.sum(burned_trace_mean)

plt.plot(RMSE, color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("RMSE")
plt.show()
plt.savefig(f"{dir_path}/{name}.png")
plt.close()