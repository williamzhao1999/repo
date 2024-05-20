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

data_directory = "data/standard_graph"

time_consumed_per_hundred_iterations = 0

f = open(data_directory+'/A.json')
A_matrix = np.array(json.load(f))
f.close()

f = open(data_directory+'/B.json')
B_matrix = np.array(json.load(f))
f.close()

f = open(data_directory+'/H.json')
H_matrix = np.array(json.load(f))
f.close()

f = open(data_directory+'/u.json')
u = np.array(json.load(f))
f.close()

a_matrix_shape = A_matrix[0].shape
b_matrix_shape = B_matrix[0].shape
h_matrix_shape = H_matrix[0].shape

x_length = a_matrix_shape[0]
y_length = h_matrix_shape[0]

f = open(data_directory+'/lambdas.json')
lambdas = np.array(json.load(f))
f.close()

N_parameters = lambdas.shape[0]

H = H_matrix
B = B_matrix
A = A_matrix

# Set the random seed to replicate results in tutorial
np.random.seed(10)

noObservations = 250
initialLambda = np.ones(N_parameters) * 0
noParticles = 251 
noBurnInIterations = 1
noIterations = 100
stepSize = np.eye(N_parameters) * (0.10**2)

cov = np.eye(y_length) * 0.05
yhatVariance = np.zeros((noParticles, y_length, y_length))
for i in range(noParticles):
    yhatVariance[i] = cov

print(f"A matrix shape: {A_matrix.shape}, B matrix shape: {B_matrix.shape}, H matrix shape: {H_matrix.shape}")

##############################################################################
# Fully-adapted particle filter for the linear Gaussian SSM
##############################################################################
def particleFilter(observations, parameters, noParticles, initialState, particleProposalDistribution, observationProposalDistribution):
        
    noObservations, dimension = observations.shape
    noObservations = noObservations - 1

    particles = np.zeros((noParticles, noObservations, x_length))
    ancestorIndices = np.zeros((noParticles, noObservations, x_length))
    weights = np.zeros((noParticles, noObservations))
    normalisedWeights = np.zeros((noParticles, noObservations))
    xHatFiltered = np.zeros((noObservations, 1, x_length))

    # Set the initial state and weights
    initialization_ancestors = range(noParticles)
    for i in range(x_length):
        ancestorIndices[:, 0, i] = initialization_ancestors

    particles[:, 0] = initialState
    xHatFiltered[0] = initialState
    normalisedWeights[:, 0] = 1.0 / noParticles
    logLikelihood = 0

    for t in range(1, noObservations):
    
        x = particles[: ,t-1]
        trans = np.matmul(A[t-1], x.T)
        u = B[t-1] @ parameters

        particles[:, t] = trans.T + u.T

        yhatMean = particles[:, t] @ H[t].T
        
        weights[:, t] = multiple_logpdfs(observations[t + 1], yhatMean, yhatVariance)

        sumWeights = logsumexp(weights[:, t])

        # Estimate log-likelihood
        predictiveLikelihood = sumWeights - np.log(noParticles)
        logLikelihood += predictiveLikelihood

    return xHatFiltered, logLikelihood

##############################################################################
# Particle Metropolis-Hastings (PMH) for the LGSS model
##############################################################################
def particleMetropolisHastings(observations, initialParameters, noParticles, 
        initialState, particleFilter, noIterations, stepSize, particleProposalDistribution, observationProposalDistribution):

    global time_consumed_per_hundred_iterations
    start_time = time.time()
    running_time = time.time()

    lambda_array = np.zeros((noIterations, N_parameters))
    lambda_proposed = np.zeros((noIterations, N_parameters))
    logLikelihood = np.zeros((noIterations))
    logLikelihoodProposed = np.zeros((noIterations))
    proposedAccepted = np.zeros((noIterations))

    initialParameters = np.array(initialParameters)
    # Set the initial parameter and estimate the initial log-likelihood
    lambda_array[0] = initialParameters
    
    _, logLikelihood[0] = particleFilter(observations, initialParameters, noParticles, initialState, particleProposalDistribution, observationProposalDistribution)
    
    for k in range(1, noIterations):
        # Propose a new parameter

        lambda_proposed[k, :] = lambda_array[k - 1, :] + multivariate_normal(mean = np.zeros(N_parameters), cov = stepSize)
        prior = 0

        _, logLikelihoodProposed[k] = particleFilter(observations, lambda_proposed[k], noParticles, initialState, particleProposalDistribution, observationProposalDistribution)

        # Compute the acceptance probability
        acceptProbability = np.min((0.0, logLikelihoodProposed[k] - logLikelihood[k - 1]))
        
        # Accept / reject step
        uniformRandomVariable = np.log(uniform())
        if uniformRandomVariable < acceptProbability:
            # Accept the parameter
            lambda_array[k] = lambda_proposed[k]
            logLikelihood[k] = logLikelihoodProposed[k]
            proposedAccepted[k] = 1.0
        else:
            # Reject the parameter
            lambda_array[k] = lambda_array[k - 1]
            logLikelihood[k] = logLikelihood[k - 1]
            proposedAccepted[k] = 0.0

        # Write out progress
        if np.remainder(k, 100) == 0:
            print("#####################################################################")
            print(" Iteration: " + str(k) + " of : " + str(noIterations) + " completed.")
            print("")
            print(" Current state of the Markov chain:")
            for i in range(N_parameters):
                print(" %.4f" % lambda_array[k, i], end = '')
            print(" Proposed next state of the Markov chain: ")
            for i in range(N_parameters):
                print(" %.4f" % lambda_proposed[k, i], end = '')
            print(" Current posterior mean:")
            for i in range(N_parameters):
                print(" %.4f" % np.mean(lambda_array[0:k, i]), end = '')
            
            print(" Current acceptance rate:                 " + "%.4f" % np.mean(proposedAccepted[0:k]) +  ".")
            print("acceptProbability %.4f, Likelihood timestep k: %.4f, Likelihood timestep k-1: %.4f, acceptance probability: %.4f, uniform: %.4f" % (acceptProbability, logLikelihoodProposed[k], logLikelihood[k - 1],
                acceptProbability, uniformRandomVariable))
            print("#####################################################################")
            if time_consumed_per_hundred_iterations == 0:
                time_consumed_per_hundred_iterations = time.time() - start_time
            
            print("Time consumed per 100 iterations: ", time_consumed_per_hundred_iterations)
    
    running_time = time.time() - running_time
    print("Total running time: ", running_time)
    return lambda_array

##############################################################################
# PMH
##############################################################################

trace_result = particleMetropolisHastings(
    observations, initialLambda, noParticles, 
    initialState, particleFilter, noIterations, stepSize)

# store result
with open(data_directory+'/result.json', 'w') as f:
    json.dump(trace_result.tolist(), f)

##############################################################################
# Plot the results
##############################################################################

burned_trace_mean = np.zeros(N_parameters)
standard_deviation = 0

dir_path = "./images"

if os.path.isdir(dir_path) == False:
    os.makedirs(dir_path) 

for t in range(N_parameters):
    trace = trace_result[noBurnInIterations:noIterations, t]
    burned_trace_mean[t] = np.sqrt(np.mean( (lambdas[t] - trace) ** 2))
    standard_deviation += np.var(trace)

    noBins = int(np.floor(np.sqrt(noIterations - noBurnInIterations)))
    grid = np.arange(noBurnInIterations, noIterations, 1)
    plot(trace, noBins, grid, lambdas[t], dir_path, f"lambda_{t}")

    trace_noburned = trace_result[:, t]
    noBins2 = int(np.floor(np.sqrt(noIterations)))
    grid2 = np.arange(0, noIterations, 1)
    plot(trace_noburned, noBins2, grid2, lambdas[t], dir_path, f"lambda_{t}_noburned")

print(f"RMSE: {np.sum(burned_trace_mean)}")
print(f"Std: {np.sqrt(standard_deviation)}")