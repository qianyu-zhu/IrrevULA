import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from helper import gradfunc, getoptJ, getnoptJ, getJ, getoptJ_JSJ, getoptJ_JSJS, getoptJ_JSSJ, initialize_A1, iterate_An, initialize_FIM, iterate_FIM


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

def simulate_chain(mm, hh, K, sigma_true):
    # Setup
    d = 4
    N = 10                                              # descretize chains
    N_stats = 8                                         # number of statistics
    np.random.seed(mm)
    mu_true = [0 for _ in range(d)]                     # true mean
    cov_true = np.diag(np.array(sigma_true)**2)         # true covariance matrix
    fisher = np.linalg.inv(cov_true)                    # Fisher information matrix
    initcond = np.random.multivariate_normal(mu_true, cov_true)                # initial condition
    fisher_error = None

    # Optimal Langevin, optimal perturbation 1
    Y = initcond
    statsY_JSJ = np.zeros((N, N_stats))
    J_opt_JSJ = getoptJ_JSJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt_JSJ) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsY_JSJ[nn] = stat_func(Y_his)


    # Optimal Langevin, optimal perturbation 2
    Y = initcond
    statsY_JSJS = np.zeros((N, N_stats))
    J_opt_JSJS = getoptJ_JSJS(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt_JSJS) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsY_JSJS[nn] = stat_func(Y_his)

    # Optimal Langevin, optimal perturbation 3
    Y = initcond
    statsY_JSSJ = np.zeros((N, N_stats))
    J_opt_JSSJ = getoptJ_JSSJ(fisher)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt_JSSJ) @ gradeval + np.sqrt(hh) * np.random.randn(d)
        Y = Y_his[:, -1]
        statsY_JSSJ[nn] = stat_func(Y_his)

    # Adaptive Langevin
    Y = initcond
    statsYadapt = np.zeros((N, N_stats))
    fisher_error = np.zeros(K)
    gradeval = gradfunc(Y, fisher)
    FIM = initialize_FIM(gradeval)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N -1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            FIM = iterate_FIM(FIM, gradeval, K//N*nn + kk+1)
            if kk % 100 == 0:
                J_opt = getoptJ(FIM)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
            fisher_error[(K//N) * nn + kk] = np.linalg.norm(fisher - FIM)
        fisher_error[(K//N) * nn + (K//N - 1)] = np.linalg.norm(fisher - FIM)
        Y = Y_his[:, -1]
        statsYadapt[nn] = stat_func(Y_his)


    return statsY_JSJ, statsY_JSJS, statsY_JSSJ, statsYadapt, fisher_error  # shape (N, N_stats), (N, N_stats), (N, N_stats), (N, N_stats), (K)
    
def main(M,h,K,sigma_true,path):
    if path and not os.path.exists(path):
        os.makedirs(path)
    fisher = np.linalg.inv(np.diag(sigma_true)**2)
    print('total number of steps: K =', K,
          'step size =', h,
          'number of chains =', M)

    ### Run simulations in parallel
    print("Running simulations in parallel...")
    start_time = time.time()
    num_cores = min(multiprocessing.cpu_count(), 16)
    print('num_cores', num_cores)

    with Parallel(n_jobs=num_cores, backend="multiprocessing") as parallel:
        results = parallel(delayed(simulate_chain)(mm, h, K, sigma_true) for mm in tqdm(range(M)))

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")


    ### Extract results from valid chains
    start_time = time.time()

    statsY_JSJ_all, statsY_JSJS_all, statsY_JSSJ_all, statsYadapt_all, fisher_errorall = [np.array(x) for x in zip(*results)] # (M, N, N_stats), (M, N, N_stats), (M, N, N_stats), (M, N, N_stats), (M, K)
    print('fisher_errorall.shape', fisher_errorall.shape)
    np.save(f'{path}/fisher_error_4D_Gaussian_K{K}_M{M}_h{h}.npy', fisher_errorall) # (M, K)
    statsY_list = np.array([statsY_JSJ_all, statsY_JSJS_all, statsY_JSSJ_all, statsYadapt_all]) # (N_methods, M, N, N_stats)
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