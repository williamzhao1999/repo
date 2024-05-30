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
from utils import multiple_logpdfs, plot, EarlyStopping, saveResults
from scipy.special import logsumexp
from scipy import stats
import os
import math

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

f = open(data_directory+'/densities.json')
observations = np.array(json.load(f))
f.close()

N_parameters = lambdas.shape[0]

H = H_matrix
B = B_matrix
A = A_matrix

np.random.seed(10)

num_observations = 250
initial_lambda = np.ones(N_parameters) * 0
num_particles = 251 
num_burn_iterations = 10000
num_iterations = 100000
stepSize = np.eye(N_parameters) * (0.10**2)

initial_state = 0

cov = np.eye(y_length) * 0.05
yhatVariance = np.zeros((num_particles, y_length, y_length))
for i in range(num_particles):
    yhatVariance[i] = cov

early_stopping = EarlyStopping(stop_after_iterations=2)

print(f"A matrix shape: {A_matrix.shape}, B matrix shape: {B_matrix.shape}, H matrix shape: {H_matrix.shape}")

##############################################################################
# Fully-adapted particle filter for the linear Gaussian SSM
##############################################################################
def particleFilter(observations, parameters, num_particles, initial_state):
        
    num_observations, dimension = observations.shape
    num_observations = num_observations - 1

    particles = np.zeros((num_particles, num_observations, x_length))
    #ancestor_indices = np.zeros((num_particles, num_observations, x_length))
    weights = np.zeros((num_particles, num_observations))
    normalisedWeights = np.zeros((num_particles, num_observations))
    xHatFiltered = np.zeros((num_observations, 1, x_length))

    #initialization_ancestors = np.arange(num_particles)

    # Use broadcasting to set the initial ancestors across all x_length dimensions
    #ancestor_indices[:, 0, :] = initialization_ancestors[:, np.newaxis]

    particles[:, 0] = initial_state
    xHatFiltered[0] = initial_state
    normalisedWeights[:, 0] = 1.0 / num_particles
    log_likelihood = 0
    

    for t in range(1, num_observations):

        newAncestors = choice(num_particles, num_particles, p=normalisedWeights[:, t - 1], replace=True)
    
        x = particles[newAncestors ,t-1]
        trans = np.matmul(A[t-1], x.T)
        u = B[t-1] @ parameters

        v = randn(num_particles, x_length) * 0.001
        particles[:, t] = (trans.T + u.T) + v

        yhatMean = particles[:, t] @ H[t].T
        
        weights[:, t] = multiple_logpdfs(observations[t + 1], yhatMean, yhatVariance)

        maxWeight  = np.max(weights[:, t])
        weights[:, t] = weights[:, t] - maxWeight
        sumWeights = logsumexp(weights[:, t])
        
        normalisedWeights[:, t] = np.exp(weights[:, t]  - sumWeights)

        # Estimate the state
        #xHatFiltered[t] = np.sum(normalisedWeights[:, t] * particles[:, t])

        # Estimate log-likelihood
        predictive_likelihood = maxWeight + sumWeights - np.log(num_particles)
        #predictiveLikelihood = sumWeights - noParticles
        log_likelihood += predictive_likelihood

    return xHatFiltered, log_likelihood

##############################################################################
# Particle Metropolis-Hastings (PMH) for the LGSS model
##############################################################################
def particleMetropolisHastings(observations, initialParameters, num_particles, 
        initial_state, particleFilter, num_iterations, stepSize):

    global time_consumed_per_hundred_iterations
    start_time = time.time()
    running_time = time.time()

    lambda_array = np.zeros((num_iterations, N_parameters))
    lambda_proposed = np.zeros((num_iterations, N_parameters))
    log_likelihood = np.zeros((num_iterations))
    log_likelihood_proposed = np.zeros((num_iterations))
    proposed_accepted = np.zeros((num_iterations))

    initialParameters = np.array(initialParameters)
    # Set the initial parameter and estimate the initial log-likelihood
    lambda_array[0] = initialParameters
    
    _, log_likelihood[0] = particleFilter(observations, initialParameters, num_particles, initial_state)
    
    
    number_iterations_completed = -1
    for k in range(1, num_iterations):
        # Propose a new parameter

        lambda_proposed[k, :] = lambda_array[k - 1, :] + multivariate_normal(mean = np.zeros(N_parameters), cov = stepSize)

        _, log_likelihood_proposed[k] = particleFilter(observations, lambda_proposed[k], num_particles, initial_state)
        
        # Assume uniform prior, so it cancels out in the acceptance ratio
        accept_probability = np.min((1.0,  np.exp(log_likelihood_proposed[k] - log_likelihood[k - 1])))
        
        # Accept / reject step
        uniform_random_variable = uniform()
        if uniform_random_variable < accept_probability:
            # Accept the parameter
            lambda_array[k] = lambda_proposed[k]
            log_likelihood[k] = log_likelihood_proposed[k]
            proposed_accepted[k] = 1.0
        else:
            # Reject the parameter
            lambda_array[k] = lambda_array[k - 1]
            log_likelihood[k] = log_likelihood[k - 1]
            proposed_accepted[k] = 0.0

        #if early_stopping.check():
        #    number_iterations_completed = k
        #    print("Maximum perfomance reached, early stopping activated")
        #    break
            
        # Write out progress
        if np.remainder(k, 100) == 0:
            
            #early_stopping.run(N_parameters, k, num_burn_iterations, num_iterations, lambda_array)

            print("#####################################################################")
            print(" Iteration: " + str(k) + " of : " + str(num_iterations) + " completed.")
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
            
            print(" Current acceptance rate:                 " + "%.4f" % np.mean(proposed_accepted[0:k]) +  ".")
            print("acceptProbability %.4f, Likelihood timestep k: %.4f, Likelihood timestep k-1: %.4f, acceptance probability: %.4f, uniform: %.4f" % (
                accept_probability, log_likelihood_proposed[k], log_likelihood[k - 1],
                accept_probability, uniform_random_variable))
            print("#####################################################################")
            if time_consumed_per_hundred_iterations == 0:
                time_consumed_per_hundred_iterations = time.time() - start_time
            
            print("Time consumed per 100 iterations: ", time_consumed_per_hundred_iterations)

    running_time = time.time() - running_time
    print("Total running time: ", running_time)
    if number_iterations_completed != -1:
        saveResults(data_directory, number_iterations_completed, log_likelihood, proposed_accepted, log_likelihood_proposed )
        return lambda_array[0:number_iterations_completed, :]
    else:
        saveResults(data_directory, k+1, log_likelihood, proposed_accepted, log_likelihood_proposed )
        return lambda_array

##############################################################################
# PMH
##############################################################################

trace_result = particleMetropolisHastings(
    observations, initial_lambda, num_particles, 
    initial_state, particleFilter, num_iterations, stepSize)

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

effective_num_iterations = trace_result.shape[0]
effective_num_burn_iterations = math.floor((effective_num_iterations*num_burn_iterations)/num_iterations)

for t in range(N_parameters):
    trace = trace_result[effective_num_burn_iterations:effective_num_iterations, t]
    burned_trace_mean[t] = np.sqrt(np.mean( (lambdas[t] - trace) ** 2))
    standard_deviation += np.var(trace)

    num_bins = int(np.floor(np.sqrt(effective_num_iterations - effective_num_burn_iterations)))
    grid = np.arange(effective_num_burn_iterations, effective_num_iterations, 1)
    plot(trace, num_bins, grid, lambdas[t], dir_path, f"lambda_{t}")

    trace_noburned = trace_result[:, t]
    num_bins2 = int(np.floor(np.sqrt(effective_num_iterations)))
    grid2 = np.arange(0, effective_num_iterations, 1)
    plot(trace_noburned, num_bins2, grid2, lambdas[t], dir_path, f"lambda_{t}_noburned")

print(f"RMSE: {np.sum(burned_trace_mean)}")
print(f"Std: {np.sqrt(standard_deviation)}")