import os
import argparse
import numpy as np

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

def simulate_chain(mm, hh, K, fisher, means, weights, covs, inv_covs):
    # Setup
    d = len(means[0])
    np.random.seed(mm)
    Y_init = initialize(means, weights, covs)

    # vanilla Langevin
    Y_his = np.zeros(K)
    Y = Y_init
    for nn in range(K):
        Y_new = Y + hh / 2 * gradfunc(Y, means, weights, covs, inv_covs) + np.sqrt(hh) * np.random.randn(d)
        Y_his[nn] = Y_new[0]
        Y = Y_new

    # RM Langevin, constant preconditioner
    Y = Y_init
    FIM_inv = np.linalg.inv(fisher)              # shape: (d, d)
    FIM_inv_sqrt = np.linalg.cholesky(FIM_inv)   # shape: (d, d)
    Yrm_his = np.zeros(K)
    for nn in range(K):
        grad_eval = gradfunc(Y, means, weights, covs, inv_covs)  # shape: (d,)
        noise = np.random.randn(d)
        drift = (hh / 2) * (FIM_inv @ grad_eval)
        diffusion = np.sqrt(hh) * (FIM_inv_sqrt @ noise)
        Y_new = Y + drift + diffusion
        Yrm_his[nn] = Y_new[0]
        Y = Y_new


    # Irreversible Langevin, random perturbation
    Y = Y_init
    J1 = getJ(fisher)*2
    Yir1_his = np.zeros(K)
    for nn in range(K):
        grad_eval = gradfunc(Y, means, weights, covs, inv_covs)
        Y_new = Y + hh / 2 * (np.eye(d) + J1) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Yir1_his[nn] = Y_new[0]
        Y = Y_new


    Y = Y_init
    J1 = getJ(fisher)*4
    Yir2_his = np.zeros(K)
    for nn in range(K):
        grad_eval = gradfunc(Y, means, weights, covs, inv_covs)
        Y_new = Y + hh / 2 * (np.eye(d) + J1) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Yir2_his[nn] = Y_new[0]
        Y = Y_new

    Y = Y_init
    J1 = getJ(fisher)*6
    Yir3_his = np.zeros(K)
    for nn in range(K):
        grad_eval = gradfunc(Y, means, weights, covs, inv_covs)
        Y_new = Y + hh / 2 * (np.eye(d) + J1) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Yir3_his[nn] = Y_new[0]
        Y = Y_new


    # Irreversible Langevin, too large perturbation
    Y = Y_init
    J_nopt = getnoptJ(fisher)
    Ynopt_his = np.zeros(K)
    for nn in range(K):
        grad_eval = gradfunc(Y, means, weights, covs, inv_covs)
        Y_new = Y + hh / 2 * (np.eye(d) + J_nopt) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Ynopt_his[nn] = Y_new[0]
        Y = Y_new


    # Optimal Langevin, optimal perturbation
    Y = Y_init
    J_opt = getoptJ(fisher)
    Yopt_his = np.zeros(K)
    for nn in range(K):
        grad_eval = gradfunc(Y, means, weights, covs, inv_covs)
        Y_new = Y + hh / 2 * (np.eye(d) + J_opt) @ grad_eval + np.sqrt(hh) * np.random.randn(d)
        Yopt_his[nn] = Y_new[0]
        Y = Y_new

    return Y_his, Yrm_his, Yir1_his, Yir2_his, Yir3_his, Ynopt_his, Yopt_his  # (N_methods, K)
    
def main(h,K,means=None,weights=None,covs=None,path=None):
    # initialize parameters
    methods = ['Standard', 'RM', 'irr1', 'irr2', 'irr3', 'nopt', 'opt']
    if means is None or weights is None or covs is None:
        # Define GMM parameters
        means = [np.array([-30, -0, -0]), np.array([-15, 0, 0]), np.array([0, -0, 0]), np.array([15, 0, 0]), np.array([30, -0, -0])]
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

    # Create output directory if needed
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Logging
    print("parameters: K =", K, "\nstep size =", h)
    print("parameters of target distribution:")
    print("means:", means)
    print("weights:", weights)
    print("covs:", covs)
    print("fisher matrix (rounded):")
    print(np.round(fisher, decimals=2))
    print("total steps:", K)
    print("Running simulations in parallel...")

    Y_his, Yrm_his, Yir1_his, Yir2_his, Yir3_his, Ynopt_his, Yopt_his = simulate_chain(0, h, K, fisher, means, weights, covs, inv_covs)
    np.save(f'{path}/traj_h{h}.npy', [Y_his, Yrm_his, Yir1_his, Yir2_his, Yir3_his, Ynopt_his, Yopt_his])
    print("trajectory saved")
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--h', type=float, default=0.04)
    parser.add_argument('--K', type=int, default=10000)
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(h=args.h, K=args.K, path=args.path)