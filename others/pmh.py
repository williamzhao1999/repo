import matplotlib.pyplot as plt
import numpy as np
import time
import json

from mpi4py import MPI
from scipy.stats import uniform
from scipy.stats import poisson
from smcpy import MVNormal, ImproperCov, AdaptiveSampler, VectorMCMCKernel, ParallelMCMC

NUM_SNAPSHOTS = 50
NUM_FEATURES = 3
TRUE_PARAMS = np.array([[2, 3.5]])
X = np.arange(NUM_FEATURES)
TRUE_COV = np.array([[0.5, 0.25, 0.005],
                     [0.25, 0.25, 0.04],
                     [0.005, 0.04, 1]])
def eval_model(theta):
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * X + b

def gen_noisy_data(eval_model, plot=True):
    y_true = eval_model(TRUE_PARAMS)
    mean = [0] * NUM_FEATURES
    noisy_data = np.tile(y_true, (NUM_SNAPSHOTS, 1))
    noisy_data += np.random.multivariate_normal(mean, TRUE_COV, NUM_SNAPSHOTS)
    if plot:
        plot_noisy_data(X, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    for i, nd in enumerate(noisy_data):
        ax.plot(x, nd, '-o', label=f'Noisy Snapshot {i}')
    ax.plot(x, y_true[0], 'k-', linewidth=2, label='True')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend(bbox_to_anchor=(1.1, 1.0))
    plt.show()

if __name__ == '__main__':

    np.random.seed(200)

    parameters_length = 2

    num_particles = 1000

    noisy_data = gen_noisy_data(eval_model, plot=True)
    idx1, idx2 = np.triu_indices(noisy_data.shape[1])
    param_order = ["a", "b"] + [f"cov{i}" for i in range(len(idx1))]
    log_like_args = [None] * len(idx1) # estimate all variances/covariances

    priors = [uniform(0., 6.), uniform(0., 6.), 
              ImproperCov(3, dof=5, S=np.eye(3))]

    comm = MPI.COMM_WORLD.Clone()
    parallel_mcmc = ParallelMCMC(eval_model, noisy_data, priors, comm, log_like_args=log_like_args, log_like_func=MVNormal)
    mcmc_kernel = VectorMCMCKernel(parallel_mcmc, param_order)
    smc = AdaptiveSampler(mcmc_kernel)

    t0 = time.time()
    steps, mll = smc.sample(num_particles, 20, progress_bar=True)

    

    if comm.Get_rank() == 0:
        print(f'total time = {time.time() - t0}')
        print(steps)
        print(f'mean vector = {steps[-1].compute_mean()}')
        with open('data.json', 'w') as f:
            json.dump(steps.tolist(), f)
        ground_truth = np.concatenate((TRUE_PARAMS.flatten(), TRUE_COV[idx1, idx2]))
        ground_truth = pd.DataFrame(ground_truth.reshape(1, -1), columns=param_order)
        ground_truth['type'] = 'True'
        samples = pd.DataFrame(steps[-1].param_dict)
        samples['type'] = 'SMC'
        data = pd.concat([samples, ground_truth], ignore_index=True)
        
        g = sns.pairplot(data[param_order[:2] + ['type']], hue='type', corner=True)
        g.map_lower(sns.kdeplot, levels=4, palette=['.2', '.2'], warn_singular=0)
        sns.mpl.pyplot.show()
        g = sns.pairplot(data[param_order[2:] + ['type']], hue='type', corner=True)
        g.map_lower(sns.kdeplot, levels=4, palette=['.2', '.2'], warn_singular=0)
        sns.mpl.pyplot.show()
