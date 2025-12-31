import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ, getJ


"""
This script is used to run the simulation for the 4-D Gaussian with different step sizes.
"""
def stat_func(Y):
    # Y: (d, K//N)
    # we compute 8 statistics of the chain
    return np.array([np.mean(np.sum(np.abs(Y), axis=0)), np.mean(np.sum(Y**2, axis=0)), \
                     np.mean(np.max(np.abs(Y), axis=0)), np.mean(np.max(Y, axis=0)), \
                     np.sum(Y[0, :] > 1)/len(Y[0, :]), np.sum(Y[0, :] > 2)/len(Y[0, :]), \
                     np.sum(Y[3, :] > 8)/len(Y[3, :]), np.sum(Y[3, :] > 16)/len(Y[3, :])])

def simulate_chain(mm, hh, K, sigma_true):
    # Setup
    d = 4
    N = 10                                              # descretize chains
    N_stats = 8                                         # number of statistics
    np.random.seed(mm)
    mu_true = [0 for _ in range(d)]                     # true mean
    cov_true = np.diag(np.array(sigma_true)**2)         # true covariance matrix
    fisher = np.linalg.inv(cov_true)                    # Fisher information matrix
    
    # norm_sum = 0
    # for _ in range(100):
    #     J = getoptJ(fisher)
    #     norm_sum += np.linalg.norm(J, 'fro')
    # opt_norm_ave = norm_sum/100
    # print('average norm of J_opt', np.round(opt_norm_ave, 2))# average norm of J_opt ~ 8.35
    opt_norm_ave = 9.13

    # vanilla Langevin
    Y = np.random.multivariate_normal(mu_true, cov_true)
    statsY = np.zeros((N, N_stats))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * gradfunc(Y_his[:, kk], fisher) + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsY[nn] = stat_func(Y_his)

    # RM Langevin, constant preconditioner
    # Y = np.random.multivariate_normal(mu_true, cov_true)
    # FIM_inv = cov_true
    # statsYrm = np.zeros((N, N_stats))
    # FIM_inv_sqrt = np.linalg.cholesky(FIM_inv)
    # for nn in range(N):
    #     Y_his = np.zeros((d, K//N))
    #     Y_his[:, 0] = Y
    #     for kk in range(K//N - 1):
    #         gradeval = gradfunc(Y_his[:, kk], fisher)
    #         Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (FIM_inv) @ gradeval + np.sqrt(hh) * FIM_inv_sqrt @ np.random.randn(d)
    #     Y = Y_his[:, -1]
    #     statsYrm[nn] = stat_func(Y_his)


    # Irreversible Langevin, random perturbation
    Y = np.random.multivariate_normal(mu_true, cov_true)
    statsYir1 = np.zeros((N, N_stats))
    J_S = getJ(np.diag(sigma_true)) * opt_norm_ave / 2
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_S) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYir1[nn] = stat_func(Y_his)

    Y = np.random.multivariate_normal(mu_true, cov_true)
    statsYir2 = np.zeros((N, N_stats))
    J_M = getJ(np.diag(sigma_true)) * opt_norm_ave
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_M) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYir2[nn] = stat_func(Y_his)

    Y = np.random.multivariate_normal(mu_true, cov_true)
    statsYir3 = np.zeros((N, N_stats))
    J_L = getJ(np.diag(sigma_true)) * opt_norm_ave * 2
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_L) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYir3[nn] = stat_func(Y_his)

    # Irreversible Langevin, too large perturbation
    Y = np.random.multivariate_normal(mu_true, cov_true)
    statsYnopt = np.zeros((N, N_stats))
    J_nopt = getnoptJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_nopt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYnopt[nn] = stat_func(Y_his)

    # Optimal Langevin, optimal perturbation
    Y = np.random.multivariate_normal(mu_true, cov_true)
    statsYopt = np.zeros((N, N_stats))
    J_opt = getoptJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsYopt[nn] = stat_func(Y_his)

    return statsY, statsYir1, statsYir2, statsYir3, statsYnopt, statsYopt # (N_methods, N, N_stats)
    
def main(M,h,K,sigma_true,path):
    num_cores = min(multiprocessing.cpu_count(), 16)
    print('num_cores', num_cores)
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)
    fisher = np.linalg.inv(np.diag(sigma_true)**2)
    print('total number of steps: K =', K,
          'step size =', h,
          'number of chains =', M)
    # Generate 5 random skew-symmetric matrices
    print('fisher', np.round(fisher, decimals=3))


    # Constants
    methods = ['unperturbed', 'rand-S', 'rand-M', 'rand-L', 'spec', 'spec-E']

    print("Running simulations in parallel...")
    start_time = time.time()
    with Parallel(n_jobs=num_cores, backend="multiprocessing") as parallel:
        results = parallel(delayed(simulate_chain)(mm, h, K, sigma_true) for mm in tqdm(range(M)))

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")

    # Extract results from valid chains
    start_time = time.time()

    statsYall, statsYir1all, statsYir2all, statsYir3all, statsYnoptall, statsYoptall = [np.array(x) for x in zip(*results)] # (M, N, N_stats)
    statsY_list = np.array([statsYall, statsYir1all, statsYir2all, statsYir3all, statsYnoptall, statsYoptall]) # (N_methods, M, N,N_stats)
    print('statsY_list.shape', statsY_list.shape)
    np.save(f'{path}/statsY_list_K{K}_M{M}_h{h}.npy', statsY_list)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.04)
    parser.add_argument('--K', type=int, default=100000)
    parser.add_argument('--sigma_true', type=float, nargs='+', default=[2**i for i in range(4)])
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(M=args.M, h=args.h, K=args.K, sigma_true=args.sigma_true, path=args.path)