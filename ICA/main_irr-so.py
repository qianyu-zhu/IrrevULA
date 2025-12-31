import os
import argparse
import numpy as np
from tqdm import tqdm
from time import time
import scipy.io as sio
from getoptJ import getnoptJ
from gradfunc import gradlogpos
from joblib import Parallel, delayed

def stat_func(Y):
    # Y: (d, K//N)
    # we compute some statistics of the chain
    return np.array([np.mean(np.sum(np.abs(Y), axis=0)), np.mean(np.sum(Y**2, axis=0)), \
                     np.mean(np.max(np.abs(Y), axis=0)), np.mean(np.max(Y, axis=0)), \
                     np.mean((Y[0, :] > 0) & (Y[1, :] > 0)), np.mean((Y[3, :] > 0) & (Y[4, :] > 0)), \
                     np.mean(np.sum(Y, axis=0)), np.mean(np.exp(np.sum(np.abs(Y), axis=0)/2))])


def main(n, dt, T, lambda_, num_chains, subsample_rate, path):
    # Seed initialization
    np.random.seed(None)
    num_cpus = min(os.cpu_count(), 8)
    N_stats = 8
    N = 20

    # Load ICA data globally (shared in memory)
    X = sio.loadmat('ica_data3.mat')['X']

    # Parameters
    d = X.shape[0]
    num_steps = int(T / dt)

    print('dt =', dt,
        'parameters: T =', T, 
        'num_steps =', num_steps,
        'num_chains =', num_chains, 
        'lambda_ =', lambda_, 'd =', d, 
        'subsample_rate =', subsample_rate
        )

    # FIM = np.load('E_FIM.npy')
    J = np.load('J_nopt.npy')
    # J = getnoptJ(FIM)

    print('Running chains...')
    def run_chain(ii, subsample_rate):
        first_entry = []
        stats = np.zeros((N, N_stats))
        W = np.diag(np.where(np.random.rand(d) > 0.5, 1, -1)).reshape(d**2)
        for nn in range(N):
            W_ir = np.zeros((d**2, num_steps//N+1))
            W_ir[:, 0] = W
            for jj in range(num_steps//N):
                noise = np.sqrt(dt) * np.random.randn(d**2)
                grad = gradlogpos(W_ir[:, jj], X, lambda_)
                W_ir[:, jj + 1] = W_ir[:, jj] + dt * (np.eye(d**2) + J) @ grad / 2 + noise


            W_ir_sub = W_ir[:,::subsample_rate]
            first_entry.append(W_ir_sub[:3]) #save the first 3 entries of the chain
            stats[nn] = stat_func(W_ir_sub)
            W = W_ir[:, -1]

        return stats, np.array(first_entry).flatten()

    # Run chains in parallel
    print(f'cpu count: {os.cpu_count()}')
    print(f'num_chains: {num_chains}')
    print(f'num cpus: {num_cpus}')
    start_time = time()
    results = Parallel(n_jobs=num_cpus, backend='loky')(
        delayed(run_chain)(i, subsample_rate) for i in tqdm(range(num_chains))
    ) # N_chains x N x N_stats

    stats, first_entry = [np.array(x) for x in zip(*results)]
    end_time = time()
    print(f'Time taken: {end_time - start_time} seconds')

    # Unpack and average results
    print('stats.shape', np.array(stats).shape)
    print('first_entry.shape', np.array(first_entry).shape)
    # Save results
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{path}/num_chains{num_chains}_dt{dt}_T{T}.npy'
    np.save(filename, {
            'stats': stats, 
            'first_entry': first_entry
            })
    print('Statistics saved to', filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--dt', type=float, default=0.5*1e-4)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--lambda_', type=float, default=1)
    parser.add_argument('--num_chains', type=int, default=25)
    parser.add_argument('--subsample_rate', type=int, default=500)
    parser.add_argument('--path', type=str, default='results/GiIrr')
    args = parser.parse_args()
    main(args.n, args.dt, args.T, args.lambda_, args.num_chains, args.subsample_rate, args.path)