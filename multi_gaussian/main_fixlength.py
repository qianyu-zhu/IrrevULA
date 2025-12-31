import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from getoptJ import getoptJ, getnoptJ, getJ
from fisher import score_log_gmm as gradfunc
from fisher import fisher_info_pi as estimate_fisher


"""
This script is used to run the simulation for the 4-D Gaussian with different step sizes.
"""


def stat_func(Y):
    # Y: (d, K//N)
    # we compute some statistics of the chain
    return np.array([np.mean(np.sum(np.abs(Y), axis=0)), np.mean(np.sum(Y**2, axis=0)), \
                     np.mean(np.max(np.abs(Y), axis=0)), np.mean(np.max(Y, axis=0)), \
                     np.sum(Y[1, :] > 0)/len(Y[1, :]), np.sum(Y[0, :] > 20)/len(Y[0, :]), \
                     np.sum(Y[0, :] > 0)/len(Y[0, :]), np.sum(Y[0, :] > -20)/len(Y[0, :])])

def initialize(means, weights, covs):
    k = np.random.choice(len(means), p=weights)
    x = np.random.multivariate_normal(means[k], covs[k])
    return x   

def simulate_chain(mm, hh, T, fisher, means, weights, covs, inv_covs):
    # Setup
    d = len(means[0])
    N = 10                                              # descretize chains
    N_stats = 8                                         # number of statistics
    K = int(T / hh)                                     # number of steps
    np.random.seed(mm)

    # vanilla Langevin
    Y = initialize(means, weights, covs)
    statsY = np.zeros((N, N_stats))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * gradfunc(Y_his[:, kk], means, weights, covs, inv_covs) + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsY[nn] = stat_func(Y_his)

    # RM Langevin, constant preconditioner
    Y = initialize(means, weights, covs)         # shape: (d,)
    FIM_inv = np.linalg.inv(fisher)              # shape: (d, d)
    FIM_inv_sqrt = np.linalg.cholesky(FIM_inv)   # shape: (d, d)
    statsYrm = np.zeros((N, N_stats))            # result container

    # Simulate N trajectories
    for nn in range(N):
        T = K // N                                # steps per chain
        Y_his = np.zeros((d, T))
        Y_his[:, 0] = Y                           # initialize trajectory

        for kk in range(T - 1):
            grad_eval = gradfunc(Y_his[:, kk], means, weights, covs, inv_covs)  # shape: (d,)
            noise = np.random.randn(d)
            drift = (hh / 2) * (FIM_inv @ grad_eval)
            diffusion = np.sqrt(hh) * (FIM_inv_sqrt @ noise)
            Y_his[:, kk + 1] = Y_his[:, kk] + drift + diffusion

        Y = Y_his[:, -1]                          # continue from last position
        statsYrm[nn] = stat_func(Y_his)          # compute stats on the path

    # Optimal Langevin, optimal perturbation
    Y = initialize(means, weights, covs)
    statsYopt = np.zeros((N, N_stats))
    J_opt = getoptJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y

        for kk in range(K//N - 1):
            grad_eval = gradfunc(Y_his[:, kk], means, weights, covs, inv_covs)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYopt[nn] = stat_func(Y_his)
        
    norm = np.linalg.norm(J_opt)
    # Irreversible Langevin, random perturbation
    Y = initialize(means, weights, covs)
    statsYir1 = np.zeros((N, N_stats))
    J1_unscaled = getJ(fisher)
    J1 = J1_unscaled * .3 * (norm / np.linalg.norm(J1_unscaled))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y

        for kk in range(K//N - 1):
            grad_eval = gradfunc(Y_his[:, kk], means, weights, covs, inv_covs)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J1) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYir1[nn] = stat_func(Y_his)

    Y = initialize(means, weights, covs)
    statsYir2 = np.zeros((N, N_stats))
    J1_unscaled = getJ(fisher)
    J1 = J1_unscaled * (norm / np.linalg.norm(J1_unscaled))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y

        for kk in range(K//N - 1):
            grad_eval = gradfunc(Y_his[:, kk], means, weights, covs, inv_covs)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J1) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYir2[nn] = stat_func(Y_his) 

    Y = initialize(means, weights, covs)
    statsYir3 = np.zeros((N, N_stats))
    J1_unscaled = getJ(fisher)
    J1 = J1_unscaled * 3 * (norm / np.linalg.norm(J1_unscaled))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            grad_eval = gradfunc(Y_his[:, kk], means, weights, covs, inv_covs)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J1) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYir3[nn] = stat_func(Y_his)


    # Irreversible Langevin, too large perturbation
    Y = initialize(means, weights, covs)
    statsYnopt = np.zeros((N, N_stats))
    J_nopt = getnoptJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            grad_eval = gradfunc(Y_his[:, kk], means, weights, covs, inv_covs)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_nopt) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYnopt[nn] = stat_func(Y_his)


    return statsY, statsYrm, statsYir1, statsYir2, statsYir3, statsYnopt, statsYopt  # (N_methods, N, N_stats)
    
def main(M,h,T,means=None,weights=None,covs=None,path=None):
    # initialize parameters
    methods = ['Unperturbed', 'rev', 'irr-S', 'irr-M', 'irr-L', 'irr-SO', 'irr-O']
    if means is None or weights is None or covs is None:
        # Define GMM parameters
        means = [np.array([-23, -0, -0]), np.array([-11, 0, 0]), np.array([0, -0, 0]), np.array([11, 0, 0]), np.array([23, -0, -0])]
        weights = [0.3, 0.1, 0.2, 0.1, 0.3]
        covs = [
            np.array([[5, 0, 0], 
                    [0, 1, 1/5], 
                    [0, 1/5, 1/5]]),
            np.array([[10, 0, 0], 
                    [0, 1, 1/5], 
                    [0, 1/5, 1/5]]),
            np.array([[5, 0, 0], 
                    [0, 1, 1/5], 
                    [0, 1/5, 1/5]]),
            np.array([[10, 0, 0], 
                    [0, 1, 1/5], 
                    [0, 1/5, 1/5]]),
            np.array([[5, 0, 0], 
                    [0, 1, 1/5], 
                    [0, 1/5, 1/5]])
        ]
    inv_covs = [np.linalg.inv(cov) for cov in covs]
    fisher = estimate_fisher(means, weights, covs)

    # CPU setup
    num_cores = min(multiprocessing.cpu_count(), 16)

    # Create output directory if needed
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Logging
    print("num_cores:", num_cores, "\n parameters: T =", T, "\n step size =", h, "\n number of chains =", M)
    print("parameters of target distribution:")
    print("means:", means)
    print("weights:", weights)
    print("covs:", covs)
    print("fisher matrix (rounded):")
    print(np.round(fisher, decimals=2))
    print("total chain length:", T)
    print("Running simulations in parallel...")

    start_time = time.time()
    with Parallel(n_jobs=num_cores, backend="multiprocessing") as parallel:
        results = parallel(delayed(simulate_chain)(mm, h, T, fisher, means, weights, covs, inv_covs) for mm in tqdm(range(M)))

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")

    # Extract results from valid chains
    burnin = int(50 / h)
    start_time = time.time()

    statsYall, statsYrmall, statsYir1all, statsYir2all, statsYir3all, statsYnoptall, statsYoptall = [np.array(x) for x in zip(*results)] # (M, N, N_stats)
    statsY_list = np.array([statsYall, statsYrmall, statsYir1all, statsYir2all, statsYir3all, statsYnoptall, statsYoptall]) # (N_methods, M, N,N_stats)
    print('statsY_list.shape', statsY_list.shape)
    np.save(f'{path}/statsY_list_T{T}_M{M}_h{h}.npy', statsY_list)
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.04)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, path=args.path)