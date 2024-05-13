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
from utils import multiple_logpdfs

data_directory = "data/standard_graph"
 

A = np.array([[0.75, 0], [0, 0.75]])
B = np.array([[0.5, 0.0], [0.0, 0.7]])
H = np.array([[0.75, 0.25], [0.25, 0.75]])
lambda_poisson = np.array([50, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 20])
epislon = np.array([1.00, 1.00])
omega = np.array([0.10, 0.10])
time_consumed_per_hundred_iterations = 0

f = open(data_directory+'/A_last.json')
A_matrix = np.array(json.load(f))
f.close()

f = open(data_directory+'/B.json')
B_matrix = np.array(json.load(f)[-1])
f.close()

f = open(data_directory+'/H.json')
H_matrix = np.array(json.load(f)[-1])
f.close()

print(f"A matrix shape: {A_matrix.shape}, B matrix shape: {B_matrix.shape}, H matrix shape: {H_matrix.shape}")
a_matrix_shape = A_matrix.shape
b_matrix_shape = B_matrix.shape
h_matrix_shape = H_matrix.shape

x_length = a_matrix_shape[0]
y_length = h_matrix_shape[0]
N_parameters = 24

H = H_matrix
B = B_matrix
A = A_matrix


 
def generateData(noObservations, initialState):
    state = np.zeros((noObservations + 1, x_length))
    observation = np.zeros((noObservations, y_length))
    state[0] = initialState
    
    for t in range(1, noObservations):
        u = np.zeros(lambda_poisson.shape[0])
        for k in range(lambda_poisson.shape[0]):
            u[k] = poisson.rvs(lambda_poisson[k], size=1)[0]
        state[t] = A @ state[t - 1].T + B @ u #+ epislon *randn()
        observation[t] = H @ state[t] #+ omega * randn()

    return(state, observation)


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
        # Resample (multinomial)
        newAncestors = choice(noParticles, noParticles, p=normalisedWeights[:, t - 1], replace=True)
        #ancestorIndices[:, 1:t - 1] = ancestorIndices[newAncestors, 1:t - 1]
        #ancestorIndices[:, t] = newAncestors

        x = particles[newAncestors ,t-1]
        trans = np.matmul(x,A)
        u = B @ parameters

        l = np.ones((1, x_length))*0.05

        v = randn(1, noParticles).T @ (np.ones((1, x_length))*0.05)
        particles[:, t] = trans + u.T + v

        yhatMean = particles[:, t] @ H.T
        
        weights[:, t] = multiple_logpdfs(observations[t + 1], yhatMean, yhatVariance)

        maxWeight = np.max(weights[:, t])
        weights[:, t] = np.exp(weights[:, t] - maxWeight)
        sumWeights = np.sum(weights[:, t])
        normalisedWeights[:, t] = weights[:, t] / sumWeights

        # Estimate the state
        #xHatFiltered[t] = np.sum(normalisedWeights[:, t] * particles[:, t])

        # Estimate log-likelihood
        predictiveLikelihood = maxWeight + np.log(sumWeights) - np.log(noParticles)
        logLikelihood += predictiveLikelihood

    return xHatFiltered, logLikelihood

##############################################################################
# Particle Metropolis-Hastings (PMH) for the LGSS model
##############################################################################
def particleMetropolisHastings(observations, initialParameters, noParticles, 
        initialState, particleFilter, noIterations, stepSize, particleProposalDistribution, observationProposalDistribution):

    global time_consumed_per_hundred_iterations
    start_time = time.time()

    lambda_array = np.zeros((noIterations, N_parameters))
    lambda_proposed = np.zeros((noIterations, N_parameters))
    logLikelihood = np.zeros((noIterations))
    logLikelihoodProposed = np.zeros((noIterations))
    proposedAccepted = np.zeros((noIterations))

    initialParameters = np.array(initialParameters)
    # Set the initial parameter and estimate the initial log-likelihood
    lambda_array[0] = initialParameters
    parameters = initialParameters.reshape(N_parameters,1)
    
    _, logLikelihood[0] = particleFilter(observations, parameters, noParticles, initialState, particleProposalDistribution, observationProposalDistribution)
    
    for k in range(1, noIterations):
        # Propose a new parameter

        lambda_proposed[k, :] = lambda_array[k - 1, :] + multivariate_normal(mean = np.zeros(N_parameters), cov = stepSize)
        prior = 0
        for i in range(N_parameters):
            prior += (gamma.logpdf(lambda_proposed[k, i], 1) - gamma.logpdf(lambda_array[k - 1, i], 1))

        parameters = lambda_proposed[k].reshape(N_parameters,1)
        _, logLikelihoodProposed[k] = particleFilter(observations, parameters, noParticles, initialState, particleProposalDistribution, observationProposalDistribution)

        #sigmav prior
        # Compute the acceptance probability
        acceptProbability = np.min((1.0, np.exp(prior + logLikelihoodProposed[k] - logLikelihood[k - 1])))
        
        # Accept / reject step
        uniformRandomVariable = uniform()
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
        #if np.remainder(k, 100) == 0:
        print("#####################################################################")
        print(" Iteration: " + str(k) + " of : " + str(noIterations) + " completed.")
        print("")
        print(" Current state of the Markov chain:       ")
        for i in range(N_parameters):
            print(" %.4f" % lambda_array[k, i], end = '')
        print(" Proposed next state of the Markov chain: ")
        for i in range(N_parameters):
            print(" %.4f" % lambda_proposed[k, i], end = '')
        print(" Current posterior mean:                  " + "%.4f")
        for i in range(N_parameters):
            print(" %.4f" % np.mean(lambda_array[0:k, i]), end = '')
        
        print(" Current acceptance rate:                 " + "%.4f" % np.mean(proposedAccepted[0:k]) +  ".")
        print("acceptProbability %.4f, Likelihood timestep k: %.4f, Likelihood timestep k-1: %.4f, division: %.4f" % (acceptProbability, logLikelihoodProposed[k], logLikelihood[k - 1],
        np.exp(logLikelihoodProposed[k] - logLikelihood[k - 1])))
        print("#####################################################################")
        if time_consumed_per_hundred_iterations == 0:
            time_consumed_per_hundred_iterations = time.time() - start_time
        
        print("Time consumed per 1 iterations: ", time_consumed_per_hundred_iterations)
    
    return lambda_array


# Set the random seed to replicate results in tutorial
np.random.seed(10)

noObservations = 250
initialState = 0

state, observations = generateData(noObservations, initialState)
# real data
T = 721
f = open(data_directory+'/densities.json')
observations = np.array(json.load(f))
f.close()

noObservations = observations.shape[0]
##############################################################################
# PMH
##############################################################################
initialLambda = np.zeros(N_parameters)

noParticles = noObservations  # Use noParticles ~ noObservations
noBurnInIterations = 100
noIterations = 10000
stepSize = np.eye(N_parameters) * 0.5

#   particleProposalDistribution(t, particles[newAncestors, t - 1], phi, lambda_poisson, B_value, sigmav, noParticles)
def particleProposalDistribution(t, particles, parameters, noParticles):
    x = particles[: ,t-1]
    #print(particls.shape, phi.shape, particls @ phi.T)
    t = np.matmul(x,A)
    u = B @ parameters

    v = randn(1, noParticles).T @ (np.ones((1, x_length))*0.05)
    #print(v.shape)
    #print(noise.shape, randn(1, noParticles).shape, noise @ randn(1, noParticles))
    #print(o.shape)
    #print(k.shape)
    x = t + u.T + v
    return x

cov = np.eye(y_length) * 0.05
yhatVariance = np.zeros((noParticles, y_length, y_length))
for i in range(noParticles):
    yhatVariance[i] = cov


#   observationProposalDistribution(t, observations, particles, H_value, sigmae)
def observationProposalDistribution(t, observations, particles):
    yhatMean = particles[:, t] @ H.T
    lpdf = np.zeros(noParticles)
    for i in range(noParticles):
        lpdf[i] = multivariate_norm.logpdf(observations[t + 1], yhatMean[i], yhatVariance[i])
    return lpdf


trace_result = particleMetropolisHastings(
    observations, initialLambda, noParticles, 
    initialState, particleFilter, noIterations, stepSize, particleProposalDistribution, observationProposalDistribution)

with open(data_directory+'/result.json', 'w') as f:
    json.dump(trace_result.tolist(), f)

print("Time consumed per 100 iterations: ", time_consumed_per_hundred_iterations)
##############################################################################
# Plot the results
##############################################################################
noBins = int(np.floor(np.sqrt(noIterations - noBurnInIterations)))
grid = np.arange(noBurnInIterations, noIterations, 1)

for t in range(N):
    trace = trace_result[noBurnInIterations:noIterations, t]

    # Plot the parameter posterior estimate (solid black line = posterior mean)
    plt.subplot(3, 1, 1)
    plt.hist(trace, noBins, density=True, facecolor='#7570B3')
    plt.xlabel("phi")
    plt.ylabel("posterior density estimate")
    plt.axvline(np.mean(trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, trace, color='#7570B3')
    plt.xlabel("iteration")
    plt.ylabel("phi")
    plt.axhline(np.mean(trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(trace - np.mean(trace), trace - np.mean(trace), mode='full')
    idx = int(macf.size/2)
    macf = macf[idx:]
    macf = macf[0:noIterations]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#7570B3')
    plt.xlabel("lag")
    plt.ylabel("ACF of phi")

    plt.show()