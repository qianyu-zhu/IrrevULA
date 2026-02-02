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
This script is used to run the simulation for the fisher convergence rate.
"""

def simulate_chain(mm, hh, K, sigma_true):
    # Setup
    d = 4
    N = 10                                              # descretize chains
    np.random.seed(mm)
    mu_true = [0 for _ in range(d)]                     # true mean
    cov_true = np.diag(np.array(sigma_true)**2)         # true covariance matrix
    fisher = np.linalg.inv(cov_true)                    # Fisher information matrix
    initcond = np.random.multivariate_normal(mu_true, cov_true)                # initial condition
    opt_norm_ave = 9.13

    Y = initcond
    J_M = getJ(np.diag(sigma_true)) * opt_norm_ave
    fisher_error1 = np.zeros((K,))
    FIM = initialize_FIM(gradfunc(Y, fisher))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_M) @ gradeval + np.sqrt(hh) * np.random.randn(d)
            FIM = iterate_FIM(FIM, gradeval, K//N*nn + kk+1)
            fisher_error1[K//N*nn + kk] = np.linalg.norm(fisher - FIM)
        fisher_error1[(K//N) * nn + (K//N - 1)] = np.linalg.norm(fisher - FIM)
        Y = Y_his[:, -1]
        

    # Irreversible Langevin, too large perturbation
    Y = initcond
    FIM = initialize_FIM(gradfunc(Y, fisher))
    J_nopt = getnoptJ(FIM)
    fisher_error2 = np.zeros((K,))
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N - 1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            if kk % 100 == 0:
                J_nopt = getnoptJ(FIM)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_nopt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
            FIM = iterate_FIM(FIM, gradeval, K//N*nn + kk+1)
            fisher_error2[K//N*nn + kk] = np.linalg.norm(fisher - FIM)
        fisher_error2[(K//N) * nn + (K//N - 1)] = np.linalg.norm(fisher - FIM)
        Y = Y_his[:, -1]


    # Adaptive Langevin
    Y = initcond
    fisher_error3 = np.zeros((K,) )
    gradeval = gradfunc(Y, fisher)
    FIM = initialize_FIM(gradeval)
    for nn in range(N):
        Y_his = np.zeros((d, K//N))
        Y_his[:, 0] = Y
        for kk in range(K//N -1):
            gradeval = gradfunc(Y_his[:, kk], fisher)
            if kk % 100 == 0:
                J_opt = getoptJ(FIM)
            Y_his[:, kk + 1] = Y_his[:, kk] + hh / 2 * (np.eye(d) + J_opt) @ gradeval + np.sqrt(hh) * np.random.randn(d)
            FIM = iterate_FIM(FIM, gradeval, K//N*nn + kk+1)
            fisher_error3[K//N*nn + kk] = np.linalg.norm(fisher - FIM)
        fisher_error3[(K//N) * nn + (K//N - 1)] = np.linalg.norm(fisher - FIM)
        Y = Y_his[:, -1]


    return fisher_error1, fisher_error2, fisher_error3  # (K), (K), (K)
    
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

    fisher_error1_all, fisher_error2_all, fisher_error3_all = [np.array(x) for x in zip(*results)] # (M, K), (M, K), (M, K)
    print('fisher_error1_all.shape', fisher_error1_all.shape)
    print('fisher_error2_all.shape', fisher_error2_all.shape)
    print('fisher_error3_all.shape', fisher_error3_all.shape)
    fisher_error_all = np.stack([fisher_error1_all, fisher_error2_all, fisher_error3_all], axis=0) # (3, M, K)
    np.save(f'{path}/fisher_convergence_4D_Gaussian_K{K}_M{M}_h{h}.npy', fisher_error_all) # (3, M, K)
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