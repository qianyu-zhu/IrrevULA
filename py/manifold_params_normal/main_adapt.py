import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ, getJ

def main(M,h,T,num_chains):
    # Set random seed
    np.random.seed(42)

    # Data
    N = 30
    mu_true = [0, 0]
    sigma_true = [5, 10]
    X = mu_true + sigma_true * np.random.randn(N, 2) # random initialization
    d = X.shape[1]*2

    # Discretization
    K = int(T / h) + 1 # number of steps
    initcond = np.array([5, 20, 5, 40]) # initial condition, true value is ~[-0.7, 4.4, -1.6, 10.3]

    # Functions
    B = lambda state: (1 / N) * np.diag([state[1] ** 2, state[1] ** 2 / 2, state[3] ** 2, state[3] ** 2 / 2]) # inverse Fisher

    fisher = np.load('statistics/fisher_information_500000.npy')
    inv_fisher = np.linalg.inv(fisher)


    print('fisher', np.round(fisher, decimals=3))
    J1, J2, J3 = [getJ(fisher) for _ in range(3)]
    J_nopt1, J_nopt2, J_nopt3 = [getnoptJ(fisher) for _ in range(3)]
    J_opt1, J_opt2, J_opt3 = [getoptJ(fisher) for _ in range(3)]

    print('Running chains...')
    start_time = time.time()
    def run_chain(mm):
        # Standard Langevin
        Y = np.zeros((d, K))
        Y[:, 0] = initcond
        for kk in range(K - 1):
            Y[:, kk + 1] = Y[:, kk] + h / 2 * gradfunc(Y[:, kk], X) + np.sqrt(h) * np.random.randn(d)

        # RM Langevin constant
        Yrm2 = np.zeros((d, K))
        Yrm2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yrm2[:, kk], X)
            Yrm2[:, kk + 1] = Yrm2[:, kk] + h / 2 * (inv_fisher @ gradeval) + sqrtm(h * inv_fisher) @ np.random.randn(d)

        # Irreversible Langevin
        Yir1 = np.zeros((d, K))
        Yir1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir1[:, kk], X)
            Yir1[:, kk + 1] = Yir1[:, kk] + h / 2 * (np.eye(d) + J1) @ gradeval + np.sqrt(h) * np.random.randn(d)

        Yir2 = np.zeros((d, K))
        Yir2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir2[:, kk], X)
            Yir2[:, kk + 1] = Yir2[:, kk] + h / 2 * (np.eye(d) + J2) @ gradeval + np.sqrt(h) * np.random.randn(d)

        Yir3 = np.zeros((d, K))
        Yir3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir3[:, kk], X)
            Yir3[:, kk + 1] = Yir3[:, kk] + h / 2 * (np.eye(d) + J3) @ gradeval + np.sqrt(h) * np.random.randn(d)


        Ynopt1 = np.zeros((d, K))
        Ynopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt1[:, kk], X)
            Ynopt1[:, kk + 1] = Ynopt1[:, kk] + h / 2 * (np.eye(d) + J_nopt1) @ gradeval + np.sqrt(h) * np.random.randn(d)

        Ynopt2 = np.zeros((d, K))
        Ynopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt2[:, kk], X)
            Ynopt2[:, kk + 1] = Ynopt2[:, kk] + h / 2 * (np.eye(d) + J_nopt2) @ gradeval + np.sqrt(h) * np.random.randn(d)

        Ynopt3 = np.zeros((d, K))
        Ynopt3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt3[:, kk], X)
            Ynopt3[:, kk + 1] = Ynopt3[:, kk] + h / 2 * (np.eye(d) + J_nopt3) @ gradeval + np.sqrt(h) * np.random.randn(d)

        # Optimal Langevin
        Yopt1 = np.zeros((d, K))
        Yopt1[:, 0] = initcond
        for kk in range(K - 1):
            Yopt1[:, kk + 1] = Yopt1[:, kk] + h / 2 * (np.eye(d) + J_opt1) @ gradfunc(Yopt1[:, kk], X) + np.sqrt(h) * np.random.randn(d)

        Yopt2 = np.zeros((d, K))
        Yopt2[:, 0] = initcond
        for kk in range(K - 1):
            Yopt2[:, kk + 1] = Yopt2[:, kk] + h / 2 * (np.eye(d) + J_opt2) @ gradfunc(Yopt2[:, kk], X) + np.sqrt(h) * np.random.randn(d)

        Yopt3 = np.zeros((d, K))
        Yopt3[:, 0] = initcond
        for kk in range(K - 1):
            Yopt3[:, kk + 1] = Yopt3[:, kk] + h / 2 * (np.eye(d) + J_opt3) @ gradfunc(Yopt3[:, kk], X) + np.sqrt(h) * np.random.randn(d)

        return Y, Yrm2, Yir1, Yir2, Yir3, Ynopt1, Ynopt2, Ynopt3, Yopt1, Yopt2, Yopt3


    burnin = int(50 / h)
    # Run chains in parallel
    results = Parallel(n_jobs=os.cpu_count(), backend='loky')(
        delayed(run_chain)(i) for i in tqdm(range(num_chains))
    )

    # Unpack and average results
    Y, Yrm2, Yir1, Yir2, Yir3, Ynopt1, Ynopt2, Ynopt3, Yopt1, Yopt2, Yopt3 = zip(*results)

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time

    # Constants
    mupost = [-0.720439004756125, -1.651146562434101]
    mu2post = [1.171262321162274, 6.348858222073647]
    sigmapost = [4.387137064233570, 10.330273538213467]
    sigma2post = [19.620296285028772, 108.785453529268580]
    firstmoment = np.sum([mupost, sigmapost])
    secondmoment = np.sum([mu2post, sigma2post])

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)
    methods = ['Standard', 'RM_const', 'irr_1', 'irr_2', 'irr_3', 'irr_4', 'irr_5', 'nopt_1', 'nopt_2', 'nopt_3', 'nopt_4', 'nopt_5', 'opt_1', 'opt_2', 'opt_3', 'opt_4', 'opt_5']
    colors = ['black', '#1a80bb', '#a00000', '#a00000', '#a00000', '#a00000', '#a00000', '#ea801c', '#ea801c', '#ea801c', '#ea801c', '#ea801c', '#f2c45f', '#f2c45f', '#f2c45f', '#f2c45f', '#f2c45f']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]