import os
import argparse
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from helper import gradlogpos
from joblib import Parallel, delayed


"""
Baseline: Langevin dynamics with gradient descent
this is the vanilla LD for ICA
"""
def main(n):
    # Seed initialization
    np.random.seed(None)

    # Load ICA data globally (shared in memory)
    X = sio.loadmat('ica_data3.mat')['X']

    # Parameters
    T = 10000
    dt = 1e-4
    lambda_ = 1
    d = X.shape[0]
    num_chains = 25
    subsample_rate = 500
    num_steps = int(T / dt)

    print('parameters: T =', T, 
        'dt =', dt, 
        'lambda_ =', lambda_, 'd =', d, 
        'num_chains =', num_chains, 
        'subsample_rate =', subsample_rate, 
        'num_steps =', num_steps)

    print('Running chains...')
    def run_chain(ii):
        W_ir = np.zeros((d**2, num_steps))
        W_ir[:, 0] = np.diag(np.where(np.random.rand(d) > 0.5, 1, -1)).reshape(d**2)
        Wold = W_ir[:, 0]

        for jj in range(num_steps - 1):
            noise = np.sqrt(dt) * np.random.randn(d**2)
            grad = gradlogpos(Wold, X, lambda_)
            Wnew = Wold + dt * grad / 2 + noise
            W_ir[:, jj + 1] = Wnew
            Wold = Wnew

        Wss_ir = W_ir[:, ::subsample_rate]
        obs1 = np.sum(Wss_ir, axis=0) # first moment
        obs2 = np.sum(Wss_ir ** 2, axis=0) # second moment
        return obs1, obs2, Wss_ir

    # Run chains in parallel
    results = Parallel(n_jobs=os.cpu_count(), backend='loky')(
        delayed(run_chain)(i) for i in tqdm(range(num_chains))
    )

    # Unpack and average results
    obs1, obs2, Wss_list = zip(*results)
    Wss_ir = Wss_list[0] # only use the first chain

    # Save results
    filename = f'statistics/baseline_{n}.npy'
    np.save(filename, {
        'obs1': obs1,
        'obs2': obs2,
        'Wss_ir': Wss_ir,
        'dt': dt
    })
    print('Statistics saved to', filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    main(args.n)