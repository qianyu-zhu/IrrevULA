import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from helper import *
from helper import *
from joblib import Parallel, delayed




def main(dt, T, K, M, path, N_stats):
    # Load data
    data = sio.loadmat('benchmarks.mat')
    set_ = data['german'][0, 0]
    train = set_['test']
    test = set_['train']
    ind = 42
    xtrain = set_['x'][train[:, ind].astype(int), :]
    ttrain = set_['t'][train[:, ind].astype(int), :]
    xtest = set_['x'][test[:, ind].astype(int), :]
    ttest = set_['t'][test[:, ind].astype(int), :]
    ttrain = (ttrain == 1).astype(int)
    ttest = (ttest == 1).astype(int)

    # Parameters
    n = 10
    N = 10
    alpha = 1
    if K is None:
        K = int(T / dt)
    if T is None:
        T = int(K * dt)
    X = xtrain.T
    d = xtrain.shape[1]
    np.random.seed(128423)
    num_cores = min(os.cpu_count(), 16)

    print('parameters: T =', T, 
        'd =', d, 
        'K =', K,
        'dt =', dt, 
        'alpha =', alpha,
        'M = ', M)

    # load fisher information matrix
    filename = f'trajectory_statistics_fisher.npy'
    data = np.load(filename, allow_pickle=True).item()
    Fisher = data['Fisher']

    def run_chain(ii):
        statsY = np.zeros((N, N_stats))
        Y = np.zeros(d)
        J_so = getnoptJ(Fisher)
        for nn in range(N):
            Y_his = np.zeros((d, K//N))
            Y_his[:, 0] = Y

            for kk in range(K//N - 1):
                grad = grad_logpos_blr(Y_his[:, kk], alpha, X, ttrain, n)
                Y_his[:, kk + 1] = Y_his[:, kk] + (np.eye(d) + J_so) @ grad * dt / 2 + np.sqrt(dt) * np.random.randn(d)
            Y = Y_his[:, -1]
            statsY[nn] = stat_func(Y_his)

        return statsY

    # Run chains in parallel
    print("Running simulations in parallel...")
    start_time = time.time()

    results = Parallel(n_jobs=num_cores)(
        delayed(run_chain)(i) for i in tqdm(range(M))
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # Unpack and average results
    statsYall = np.array(list(zip(*results)))
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{path}/stats_dt{dt}_T{T}_K{K}.npy'
    np.save(filename, statsYall)
    print('Statistics saved to', filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=0.5*1e-4)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--M', type=int, default=128)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--path', type=str, default='statistics/')
    parser.add_argument('--N_stats', type=int, default=8)
    args = parser.parse_args()
    main(args.dt, args.T, args.K, args.M, args.path, args.N_stats)

