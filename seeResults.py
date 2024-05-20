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

for t in range(N_parameters):
    trace = results[noBurnInIterations:noIterations, t]
    burned_trace_mean[t] = np.sqrt(np.mean( (lambdas[t] - trace) ** 2))
    standard_deviation += np.var(trace)

print(f"RMSE: {np.sum(burned_trace_mean)}")
print(f"Std: {np.sqrt(standard_deviation)}")