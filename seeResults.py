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

f = open('./result_10000.json')
results = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/lambdas.json')
lambdas = np.array(json.load(f))
f.close()

N_parameters = lambdas.shape[0]

noBurnInIterations = 10000
noIterations = 100000

burned_trace_mean = np.zeros(N_parameters)
var = 0

dir_path = "./images"

if os.path.isdir(dir_path + "/" + str(noIterations) + "/") == False:
    os.makedirs(dir_path + "/" + str(noIterations) + "/") 

if os.path.isdir(dir_path + "/best/") == False:
    os.makedirs(dir_path + "/best/") 

lambdas_results = np.zeros(N_parameters)

for t in range(N_parameters):
    trace = results[noBurnInIterations:noIterations, t]
    burned_trace_mean[t] = np.sqrt(np.mean( (lambdas[t] - trace) ** 2))
    var += np.var(trace, ddof=1)

    noBins = int(np.floor(np.sqrt(noIterations - noBurnInIterations)))
    grid = np.arange(noBurnInIterations, noIterations, 1)
    plot(trace, noBins, grid, lambdas[t], dir_path + "/" + str(noIterations) + "/", f"lambda_{t}")

    lambdas_results[t] = np.mean(trace)

    trace_noburned = results[:, t]
    noBins2 = int(np.floor(np.sqrt(noIterations)))
    grid2 = np.arange(0, noIterations, 1)
    plot(trace_noburned, noBins2, grid2, lambdas[t], dir_path + "/" + str(noIterations) + "/", f"lambda_{t}_noburned", False)

with open(dir_path+ "/" + str(noIterations) + "/" + f'/lambdas.json', 'w') as f:
    json.dump(lambdas_results.tolist(), f)
print(f"RMSE: {np.sum(burned_trace_mean)}")
print(f"Std: {np.sqrt(var)}")

RMSE = np.zeros(noIterations)
VAR = np.zeros(noIterations)
iteration_min = None
current_min_rmse = float('inf')
current_min_var = float('inf')


for i in range(noIterations):

    no_burn_iterations = math.floor((i*noBurnInIterations)/noIterations)
    trace = results[no_burn_iterations:i]
    
    variance = np.sum(np.var(trace,axis=0, ddof=1))
    rmse = np.sum(np.sqrt(np.mean( (lambdas - trace) ** 2,axis=0)))

    RMSE[i] = rmse
    VAR[i] = variance

    if RMSE[i] < current_min_rmse:
        current_min_rmse = RMSE[i]
        current_min_var = VAR[i]
        trace_min = i

lambdas_results = np.zeros(N_parameters)
for t in range(N_parameters):
    no_burn_iterations = math.floor((trace_min*noBurnInIterations)/noIterations)
    trace = results[no_burn_iterations:trace_min, t]
    noBins = int(np.floor(np.sqrt(trace_min - no_burn_iterations)))
    grid = np.arange(no_burn_iterations, trace_min, 1)
    plot(trace, noBins, grid, lambdas[t], dir_path + "/best/", f"lambda_{t}")

    lambdas_results[t] = np.mean(trace)

    trace_noburned = results[:trace_min, t]
    noBins2 = int(np.floor(np.sqrt(trace_min)))
    grid2 = np.arange(0, trace_min, 1)
    plot(trace_noburned, noBins2, grid2, lambdas[t], dir_path + "/best/", f"lambda_{t}_noburned", False)

with open(dir_path+f'/best/lambdas.json', 'w') as f:
    json.dump(lambdas_results.tolist(), f)

plt.plot(RMSE, color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("RMSE")
plt.savefig(f"{dir_path}/RMSE.png")
plt.close()

plt.plot(VAR, color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("Variance")
plt.savefig(f"{dir_path}/VAR.png")
plt.close()

plt.plot(np.sqrt(VAR), color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("STD")
plt.savefig(f"{dir_path}/STD.png")
plt.close()