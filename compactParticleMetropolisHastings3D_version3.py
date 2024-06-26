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
from utils import multiple_logpdfs
from scipy.special import logsumexp

A = np.array([[0.00028499273280664547, 5.878122876425919e-05, 0.0, 0.0, 0.0, 0.00010197564273771084, 0.00024896816516678093, 0.0, 0.0, 0.00023689944643100146, 0.011245010235017505, 0.0],
             [5.878122876425913e-05, 1.2123933200010352e-05, 0.0, 0.0, 0.0, 2.1033005035305824e-05, 5.1350974909298284e-05, 0.0, 0.0, 4.886173909646562e-05, 0.0023193416638081594, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.00010197564273771075, 2.1033005035305828e-05, 0.0, 0.0, 0.0, 3.6488760991756576e-05, 8.908538970127382e-05, 0.0, 0.0, 8.476698011243442e-05, 0.004023671533709041, 0.0], [0.0005450924732055322, 0.0001124281487798803, 0.0, 0.0, 0.0, 0.00019504411484181684, 0.0004761899419810032, 0.0, 0.0, 0.00045310665954316216, 0.021507813128645102, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0005931403624393083, 0.00012233827505178966, 0.0, 0.0, 0.0, 0.00021223653353457305, 0.0005181643274499334, 0.0, 0.0, 0.0004930463388801162, 0.023403647457064164, 0.0], [0.025374749564744684, 0.005233673660068371, 0.0, 0.0, 0.0, 0.009079552207139473, 0.022167248892578588, 0.0, 0.0, 0.021092692666278192, 1.001216121395452, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
B = np.array([[0.005555555555555556, 0.0, 0.0], [0.0, 0.005555555555555556, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.005555555555555556], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
H = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
lambda_poisson = np.array([1, 50, 20])
epislon = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
omega = np.array([0.10, 0.10, 0.10, 0.10])

data_directory = "data/mini_graph"
 
lambda_poisson = np.array([50, 10, 15])

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

A = A_matrix
B = B_matrix
H = H_matrix

time_consumed_per_hundred_iterations = 0
# Set the random seed to replicate results in tutorial
np.random.seed(10)

noObservations = 250
initialState = 0

initialLambda = [0, 0, 0]

noParticles = 251           # Use noParticles ~ noObservations
noBurnInIterations = 2000
noIterations = 10000
stepSize = np.diag((0.10**2, 0.10**2, 0.10**2))

N = 3

 
def generateData(noObservations, initialState):
    state = np.zeros((noObservations + 1, 12))
    observation = np.zeros((noObservations, 4))
    state[0] = initialState
    
    for t in range(1, noObservations):
        state[t] = state[t - 1] @ A + B @ lambda_poisson #+ epislon *randn()
        observation[t] = H @ state[t] #+ omega * randn()

    return(state, observation)

state, observations = generateData(noObservations, initialState)

with open('obs.json', 'w') as f:
    json.dump(observations.tolist(), f)

f = open(data_directory+'/obs.json')
observations2 = np.array(json.load(f))
f.close()

cov = np.eye(4) * 0.05
yhatVariance = np.zeros((noParticles, 4, 4))
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

    particles = np.zeros((noParticles, noObservations, 12))
    ancestorIndices = np.zeros((noParticles, noObservations, 12))
    weights = np.zeros((noParticles, noObservations))
    normalisedWeights = np.zeros((noParticles, noObservations))
    xHatFiltered = np.zeros((noObservations, 1, 12))

    # Set the initial state and weights
    initialization_ancestors = np.array([range(noParticles),
    range(noParticles),range(noParticles),range(noParticles),
    range(noParticles),range(noParticles),range(noParticles)
    ,range(noParticles),range(noParticles),range(noParticles)
    ,range(noParticles),range(noParticles)]).T
    ancestorIndices[:, 0, :] = initialization_ancestors
    particles[:, 0] = initialState
    xHatFiltered[0] = initialState
    normalisedWeights[:, 0] = 1.0 / noParticles
    logLikelihood = 0

    for t in range(1, noObservations):
        # Resample (multinomial)
        #newAncestors = choice(noParticles, noParticles, p=normalisedWeights[:, t - 1], replace=True)
        #ancestorIndices[:, 1:t - 1] = ancestorIndices[newAncestors, 1:t - 1]
        #ancestorIndices[:, t] = newAncestors

        x = particles[: ,t-1]
        trans = np.matmul(x,A)
        u = B @ parameters

        v = randn(noParticles, 12) * epislon
        particles[:, t] = trans + u.T
        #particles[:, t] = particles[:, t] * v

        yhatMean = particles[:, t] @ H.T
        
        weights[:, t] = multiple_logpdfs(observations[t + 1], yhatMean, yhatVariance)


        sumWeights = logsumexp(weights[:, t])
        #normalisedWeights[:, t] = weights[:, t] - sumWeights

        # Estimate the state
        #xHatFiltered[t] = np.sum(normalisedWeights[:, t] * particles[:, t])

        # Estimate log-likelihood
        predictiveLikelihood = sumWeights - np.log(noParticles)
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

    lambda_array = np.zeros((noIterations, 3))
    lambda_proposed = np.zeros((noIterations, 3))
    logLikelihood = np.zeros((noIterations))
    logLikelihoodProposed = np.zeros((noIterations))
    proposedAccepted = np.zeros((noIterations))

    initialParameters = np.array(initialParameters)
    # Set the initial parameter and estimate the initial log-likelihood
    lambda_array[0] = initialParameters
    
    _, logLikelihood[0] = particleFilter(observations, initialParameters, noParticles, initialState)
    
    start_time = time.time()
    for k in range(1, noIterations):
        # Propose a new parameter
        

        lambda_proposed[k, :] = lambda_array[k - 1, :] + multivariate_normal(mean = np.zeros(3), cov = stepSize)
        prior = 0
        #for i in range(N):
        #    prior += (gamma.logpdf(lambda_proposed[k, i], 1) - gamma.logpdf(lambda_array[k - 1, i], 1))


        _, logLikelihoodProposed[k] = particleFilter(observations, lambda_proposed[k], noParticles, initialState)

        #sigmav prior
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
            print(" Current state of the Markov chain:       " + "%.4f" % lambda_array[k,0] +  ", %.4f." %lambda_array[k,1] +", %.4f." %lambda_array[k,2])
            print(" Proposed next state of the Markov chain: " + "%.4f" % lambda_proposed[k, 0] +  ", %.4f." %lambda_proposed[k,1]+", %.4f." %lambda_proposed[k,2])
            print(" Current posterior mean:                  " + "%.4f" % np.mean(lambda_array[0:k, 0]) +  ", %.4f." % np.mean(lambda_array[0:k, 1]) +", %.4f." % np.mean(lambda_array[0:k, 2]))
            print(" Current acceptance rate:                 " + "%.4f" % np.mean(proposedAccepted[0:k]) +  ".")
            
            print("acceptProbability %.4f, Likelihood timestep k: %.4f, Likelihood timestep k-1: %.4f, uniform: %.4f" % (acceptProbability, logLikelihoodProposed[k], logLikelihood[k - 1],
                uniformRandomVariable))
            
            if time_consumed_per_hundred_iterations == 0:
                time_consumed_per_hundred_iterations = time.time() - start_time
            print("Time consumed per 100 iterations: ", time_consumed_per_hundred_iterations)
            print("#####################################################################")

    running_time = time.time() - running_time
    print("Total running time: ", running_time)
    return lambda_array




#   particleProposalDistribution(t, particles[newAncestors, t - 1], phi, lambda_poisson, B_value, sigmav, noParticles)
def particleProposalDistribution(t, particles, parameters, noParticles):
    x = particles[: ,t-1]
    #print(particls.shape, phi.shape, particls @ phi.T)
    t = np.matmul(x,A)
    u = B @ parameters

    v = randn(1, noParticles) * epislon.reshape(N,1)
    #print(v.shape)
    #print(noise.shape, randn(1, noParticles).shape, noise @ randn(1, noParticles))
    #print(o.shape)
    #print(k.shape)
    x = t + u.T + v.T
    return x


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