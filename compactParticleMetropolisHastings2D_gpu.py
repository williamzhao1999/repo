from __future__ import print_function, division
import matplotlib.pylab as plt
import numpy as np
from numpy.random import randn, choice, uniform, multivariate_normal
from scipy.stats import poisson
from scipy.stats import norm
import json
import time
from scipy.stats import gamma
from scipy.stats import multivariate_normal as multivariate_norm
from utils import multiple_logpdfs_gpu
import torch
import math
import time


np.random.seed(10)
torch.manual_seed(0)
torch.set_default_device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = "cuda:0"
pi2 = torch.tensor(2 * math.pi, dtype=torch.float32).to(device=device)
one = torch.tensor(1.0, dtype=torch.float32).to(device=device)
zero = torch.tensor(0.0, dtype=torch.float32).to(device=device)

with torch.device(device):
    A = torch.FloatTensor([[0.75, 0], [0, 0.75]]).to(device=device)
    B = torch.FloatTensor([[0.5, 0.0], [0.0, 0.7]]).to(device=device)
    H = torch.FloatTensor([[0.75, 0.25], [0.25, 0.75]]).to(device=device)
    lambda_poisson = torch.FloatTensor([1, 50]).to(device=device)
    epislon = torch.FloatTensor([1.00, 1.00]).to(device=device)
    omega = torch.FloatTensor([0.10, 0.10]).to(device=device)

    time_consumed_per_hundred_iterations = torch.tensor(0).to(device=device)
    # Set the random seed to replicate results in tutorial

    noObservations = torch.tensor(250).to(device=device)
    initialState = torch.tensor(0).to(device=device)

    initialLambda = torch.FloatTensor([0, 0]).to(device=device)

    noParticles = torch.tensor(251).to(device=device)
    noBurnInIterations = torch.tensor(2000).to(device=device)
    noIterations = torch.tensor(10000).to(device=device)
    
    stepSize = torch.diag(torch.FloatTensor([0.10**2, 0.10**2])).to(device=device)

    N = torch.tensor(2, dtype=torch.int32).to(device=device)

    
    def generateData(noObservations, initialState):
        state = torch.zeros((noObservations + 1, N), dtype=torch.float32).to(device=device)
        observation = torch.zeros((noObservations, N), dtype=torch.float32).to(device=device)
        state[0] = initialState
        
        for t in range(1, noObservations):
            u = torch.zeros(lambda_poisson.shape[0], dtype=torch.float32).to(device=device)
            for k in range(lambda_poisson.shape[0]):
                u[k] = torch.poisson(lambda_poisson[k]).to(device=device)
            state[t] = A @ state[t - 1].T + B @ u #+ epislon *randn()
            observation[t] = H @ state[t] #+ omega * randn()

        return(state, observation)

    state, observations = generateData(noObservations, initialState)

    cov = torch.eye(2) * 0.05
    yhatVariance = torch.zeros((noParticles, 2, 2))
    for i in range(noParticles):
        yhatVariance[i] = cov
    ##############################################################################
    # PMH
    ##############################################################################





    ##############################################################################
    # Fully-adapted particle filter for the linear Gaussian SSM
    ##############################################################################
    def particleFilter(observations, parameters, noParticles, initialState):

        
            
        noObservations, dimension = observations.shape
        noObservations = noObservations - 1

        particles = torch.zeros((noParticles, noObservations, dimension), dtype=torch.float32).to(device=device)
        ancestorIndices = torch.zeros((noParticles, noObservations, dimension)).to(device=device)
        weights = torch.zeros((noParticles, noObservations)).to(device=device)
        normalisedWeights = torch.zeros((noParticles, noObservations), dtype=torch.float32).to(device=device)
        xHatFiltered = torch.zeros((noObservations, 1, dimension), dtype=torch.float32).to(device=device)

        # Set the initial state and weights
        initialization_ancestors = torch.tensor([range(noParticles),range(noParticles)]).T.to(device=device)
        ancestorIndices[:, 0, :] = initialization_ancestors
        particles[:, 0] = initialState
        xHatFiltered[0] = initialState
        normalisedWeights[:, 0] = one.clone().detach() / noParticles
        logLikelihood = one.clone().detach()
        
        for t in range(1, noObservations):
            # Resample (multinomial)
            
            newAncestors = normalisedWeights[:, t - 1].multinomial(num_samples=noParticles, replacement=True).to(device=device)

            #newAncestors = choice(noParticles, noParticles, p=normalisedWeights[:, t - 1], replace=True)
            #ancestorIndices[:, 1:t - 1] = ancestorIndices[newAncestors, 1:t - 1]
            #ancestorIndices[:, t] = newAncestors

            x = particles[newAncestors ,t-1]
            trans = torch.matmul(x,A)
            u = torch.matmul(B,parameters)


            v = torch.randn(1, noParticles) * epislon.reshape(N,1)
            particles[:, t] = trans + u.T + v.T

            yhatMean = torch.matmul(particles[:, t],H.T)
            
            
            weights[:, t] = multiple_logpdfs_gpu(observations[t + 1], yhatMean, yhatVariance, device, pi2)


            maxWeight = torch.max(weights[:, t])
            weights[:, t] = torch.exp(weights[:, t] - maxWeight)
            sumWeights = torch.sum(weights[:, t])
            normalisedWeights[:, t] = weights[:, t] / sumWeights

            # Estimate the state
            #xHatFiltered[t] = np.sum(normalisedWeights[:, t] * particles[:, t])

            # Estimate log-likelihood
            predictiveLikelihood = maxWeight + torch.log(sumWeights) - torch.log(noParticles)
            #predictiveLikelihood = sumWeights - noParticles
            logLikelihood += predictiveLikelihood

        return xHatFiltered, logLikelihood

    ##############################################################################
    # Particle Metropolis-Hastings (PMH) for the LGSS model
    ##############################################################################
    def particleMetropolisHastings(observations, initialParameters, noParticles, 
            initialState, particleFilter, noIterations, stepSize):

        global time_consumed_per_hundred_iterations
        running_time = time.time()

        lambda_array = torch.zeros((noIterations, observations.shape[1]), dtype=torch.float32).to(device=device)
        lambda_proposed = torch.zeros((noIterations, observations.shape[1]), dtype=torch.float32).to(device=device)
        logLikelihood = torch.zeros((noIterations), dtype=torch.float32).to(device=device)
        logLikelihoodProposed = torch.zeros((noIterations), dtype=torch.float32).to(device=device)
        proposedAccepted = torch.zeros((noIterations)).to(device=device)

        initialParameters = torch.tensor(initialParameters, dtype=torch.float32).to(device=device)
        # Set the initial parameter and estimate the initial log-likelihood
        lambda_array[0] = initialParameters
        parameters = initialParameters.reshape(N,1)
        
        _, logLikelihood[0] = particleFilter(observations, parameters, noParticles, initialState)

        mean = torch.zeros(2).to(device=device)
        m = torch.distributions.MultivariateNormal(mean, stepSize)
        rate_gamma = torch.tensor(1).to(device=device)
        m_uniform = torch.distributions.uniform.Uniform(torch.tensor([0.0]).to(device=device), torch.tensor([1.0]).to(device=device))
        
        start_time = time.time()
        for k in range(1, noIterations):
            # Propose a new parameter
            

            lambda_proposed[k, :] = lambda_array[k - 1, :] + m.sample()
            #prior = 0
            #for i in range(N):
            #    prior += (gamma.logpdf(lambda_proposed[k, i], 1) - gamma.logpdf(lambda_array[k - 1, i], 1))

            parameters = lambda_proposed[k].reshape(N,1)
            _, logLikelihoodProposed[k] = particleFilter(observations, parameters, noParticles, initialState)

            #sigmav prior
            # Compute the acceptance probability prior + 
            acceptProbability = torch.min(zero, logLikelihoodProposed[k] - logLikelihood[k - 1])
            
            # Accept / reject step
            uniformRandomVariable = torch.log(m_uniform.sample())
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
                print(" Current state of the Markov chain:       " + "%.4f" % lambda_array[k,0] +  ", %.4f." %lambda_array[k,1])
                print(" Proposed next state of the Markov chain: " + "%.4f" % lambda_proposed[k, 0] +  ", %.4f." %lambda_proposed[k,1])
                print(" Current posterior mean:                  " + "%.4f" % torch.mean(lambda_array[0:k, 0]) +  ", %.4f." % torch.mean(lambda_array[0:k, 1]))
                print(" Current acceptance rate:                 " + "%.4f" % torch.mean(proposedAccepted[0:k]) +  ".")
                
                print("acceptProbability %.4f, Likelihood timestep k: %.4f, Likelihood timestep k-1: %.4f, uniform: %.4f" % (acceptProbability, logLikelihoodProposed[k], logLikelihood[k - 1],
                    uniformRandomVariable))
                
                if time_consumed_per_hundred_iterations == 0:
                    time_consumed_per_hundred_iterations = time.time() - start_time
                print("Time consumed per 100 iterations: ", time_consumed_per_hundred_iterations)
                print("#####################################################################")

        running_time = time.time() - running_time
        print("Total running time: ", running_time)
        return lambda_array



    trace_result = particleMetropolisHastings(
        observations, initialLambda, noParticles, 
        initialState, particleFilter, noIterations, stepSize)

    with open('data.json', 'w') as f:
        json.dump(trace_result.tolist(), f)

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

    def observationProposalDistribution():
        pass
    def particleProposalDistribution():
        pass