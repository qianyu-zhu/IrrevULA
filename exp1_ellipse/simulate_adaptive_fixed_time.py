import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from helper import gradfunc, getoptJ, getnoptJ, getJ, initialize_A1, iterate_An, initialize_FIM, iterate_FIM


"""
This script is used to run the simulation for the 4-D Gaussian with different step sizes.
"""

def stat_func(Y):
    # Y: (d, K//N)
    # we compute some statistics of the chain
    return np.array([np.mean(np.sum(np.abs(Y), axis=0)), np.mean(np.sum(Y**2, axis=0)), \
                     np.mean(np.max(np.abs(Y), axis=0)), np.mean(np.max(Y, axis=0)), \
                     np.sum(Y[0, :] > 1)/len(Y[0, :]), np.sum(Y[0, :] > 2)/len(Y[0, :]), \
                     np.sum(Y[3, :] > 8)/len(Y[3, :]), np.sum(Y[3, :] > 16)/len(Y[3, :])])

def simulate_chain(mm, hh, T, sigma_true):
    # Setup
    d = 4
    N = 10                                              # descretize chains
    N_stats = 8                                         # number of statistics
    np.random.seed(mm)
    K = int(T / hh)                                     # number of steps
    mu_true = [0 for _ in range(d)]                     # true mean
    cov_true = np.diag(np.array(sigma_true)**2)         # true covariance matrix
    fisher = np.linalg.inv(cov_true)                    # Fisher information matrix
    initcond = np.random.multivariate_normal(mu_true, cov_true)                # initial condition
    fisher_error = None

    # optimal langevin
    Y = initcond
    statsY = np.zeros((N, N_stats))
    J_opt = getoptJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsY[nn] = stat_func(Y_his)

 
    # Optimal Langevin
    Y = initcond
    statsYadapt = np.zeros((N, N_stats))
    fisher_error = np.zeros(K)
    gradeval = gradfunc(Y, fisher)
    FIM = initialize_FIM(gradeval)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            FIM = iterate_FIM(FIM, gradeval, K//N*nn + kk+1)
            if kk//100 == 0:
                J_opt = getoptJ(FIM)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
            fisher_error[K//N*nn + kk] = np.linalg.norm(fisher - FIM)
        Y = Y_his[:, -1]
        statsYadapt[nn] = stat_func(Y_his)

    return statsY, statsYadapt, fisher_error  # shape (N, N_stats), (N, N_stats), (K)



def main(M,h,T,sigma_true,path):
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Setup
    d = 4
    N = 10                                              # descretize chains
    N_stats = 8                                         # number of statistics
    K = int(T / h)                                     # number of steps
    mu_true = [0 for _ in range(d)]
    cov_true = np.diag(sigma_true)**2
    fisher = np.linalg.inv(cov_true)
    initcond = np.random.multivariate_normal(mu_true, cov_true)                # initial 
    fisher_error = None

    print('parameters: T =', T,
          'step size =', h,
          'd =', d,
          'number of chains =', M)
    # Generate 5 random skew-symmetric matrices
    print('fisher', np.round(fisher, decimals=3))


    print("Running simulations in parallel...")
    start_time = time.time()
    num_cores = 16                           # multiprocessing.cpu_count()
    print('num_cores', num_cores)
    results = Parallel(n_jobs=num_cores)(
            delayed(simulate_chain)(mm, h, T, sigma_true) for mm in tqdm(range(M))
            )

    # Extract results from valid chains
    statsYall, statsYadaptall, fisher_errorall = [np.array(x) for x in zip(*results)] # (M, N, N_stats), (M, N, N_stats), (M, K)
    print('statsYall.shape', statsYall.shape)
    print('statsYadaptall.shape', statsYadaptall.shape)
    print('fisher_errorall.shape', fisher_errorall.shape)
    np.save(f'{path}/fisher_error_4D_Gaussian_T{T}_M{M}_h{h}.npy', fisher_errorall) # (M, K)
    np.save(f'{path}/statsY_4D_Gaussian_T{T}_M{M}_h{h}.npy', statsYall) # (M, N, N_stats)
    np.save(f'{path}/statsYadapt_4D_Gaussian_T{T}_M{M}_h{h}.npy', statsYadaptall) # (M, N, N_stats)
    exit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.04)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--sigma_true', type=float, nargs='+', default=[2**i for i in range(4)])
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, sigma_true=args.sigma_true, path=args.path)