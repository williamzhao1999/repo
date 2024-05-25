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

f = open('./result_1000.json')
results = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/lambdas.json')
lambdas = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/log_likelihood_proposed.json')
log_likelihood_proposed = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/log_likelihood.json')
log_likelihood = np.array(json.load(f))
f.close()

f = open('./data/standard_graph/acceptance_rate.json')
acceptance_rate = np.array(json.load(f))
f.close()





N_parameters = lambdas.shape[0]

num_iterations = results.shape[0]
num_burn_iterations = math.floor((num_iterations*10000)/100000)

results = results[0:num_iterations,:]

burned_trace_mean = np.zeros(N_parameters)
var = 0

dir_path = "./images"

if os.path.isdir(dir_path + "/" + str(num_iterations) + "/") == False:
    os.makedirs(dir_path + "/" + str(num_iterations) + "/") 

if os.path.isdir(dir_path + "/best/") == False:
    os.makedirs(dir_path + "/best/") 

lambdas_results = np.zeros(N_parameters)

for t in range(N_parameters):
    trace = results[num_burn_iterations:num_iterations, t]
    lambdas_results[t] = np.mean(trace)
    burned_trace_mean[t] = (lambdas[t] - lambdas_results[t]) ** 2
    
    noBins = int(np.floor(np.sqrt(num_iterations - num_burn_iterations)))
    grid = np.arange(num_burn_iterations, num_iterations, 1)
    plot(trace, noBins, grid, lambdas[t], dir_path + "/" + str(num_iterations) + "/", f"lambda_{t}")

    
    # no burned
    # trace_noburned = results[:, t]
    # noBins2 = int(np.floor(np.sqrt(num_iterations)))
    # grid2 = np.arange(0, num_iterations, 1)
    # plot(trace_noburned, noBins2, grid2, lambdas[t], dir_path + "/" + str(num_iterations) + "/", f"lambda_{t}_noburned", False)

var = np.var(lambdas_results, ddof=1)
with open(dir_path+ "/" + str(num_iterations) + "/" + f'/lambdas.json', 'w') as f:
    json.dump(lambdas_results.tolist(), f)
print(f"RMSE: {np.sqrt(np.mean(burned_trace_mean))}")
print(f"Std: {np.sqrt(var)}")

RMSE = np.zeros(num_iterations)
VAR = np.zeros(num_iterations)
iteration_min = None
current_min_rmse = float('inf')
current_min_var = float('inf')


for i in range(num_iterations):

    no_burn_iterations = math.floor((i*num_burn_iterations)/num_iterations)
    trace = results[no_burn_iterations:i]
    lambda_mean = np.mean(trace,axis=0)
    
    variance = np.var(lambda_mean, ddof=1)
    rmse = np.sum(np.sqrt(np.mean( (lambdas - lambda_mean) ** 2)))

    RMSE[i] = rmse
    VAR[i] = variance

    if RMSE[i] < current_min_rmse:
        current_min_rmse = RMSE[i]
        current_min_var = VAR[i]
        trace_min = i

lambdas_results = np.zeros(N_parameters)
for t in range(N_parameters):
    no_burn_iterations = math.floor((trace_min*num_burn_iterations)/num_iterations)
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

plt.plot(log_likelihood_proposed, color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("Log Likelihood Proposed")
plt.savefig(f"{dir_path}/log_likelihood_proposed.png")
plt.close()

plt.plot(log_likelihood, color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("Log Likelihood")
plt.savefig(f"{dir_path}/log_likelihood.png")
plt.close()

plt.plot(acceptance_rate, color='#7570B3')
plt.xlabel("iteration")
plt.ylabel("Acceptance Rate")
plt.savefig(f"{dir_path}/acceptance_rate.png")
plt.close()