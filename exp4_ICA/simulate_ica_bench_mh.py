import os
import argparse
import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import sqrtm
from tqdm import tqdm
from time import time
from numpy.linalg import cholesky
from joblib import Parallel, delayed
import scipy.io as sio
from helper import *

# ------------------------------------------------------------------
# 1.  Log–posterior   log π(W | X)   (up to a constant)
# ------------------------------------------------------------------
def logpi(W_vec, X, lambd=1.0):
    """
    Log-posterior for the ICA model used by Welling & Teh (2011).

    log π(W|X) = log|det W|
               + Σ_{n,i} 2 log sech( ½ w_i^T x_n )
               – (λ/2) ||W||_F²               + const
    """
    d, N = X.shape
    W = W_vec.reshape(d, d)

    # log|det W|
    sign, logdet = slogdet(W)
    if sign <= 0:                        # singular or negative det → -∞
        return -np.inf

    # likelihood term   p(y) ∝ sech²(½y)
    Y = W @ X                            # shape (d, N)
    loglik = 2.0 * np.sum(-np.log(2*np.cosh(0.5*Y)))   # drop −log4 constant

    # Gaussian prior N(0, λ⁻¹)
    logprior = -0.5 * lambd * np.sum(W**2)

    return logdet + loglik + logprior    # constant term ignored
# ------------------------------------------------------------------


# -------------------------------------------------------------
# irmala_step.py
# -------------------------------------------------------------

def irmala_step(W_vec, X, J, lambd, dt):
    """
    Metropolis-adjusted Riemannian Langevin step with irreversible drift J.
    Metric G(W) = I + WᵀW.
    """
    d, _ = X.shape
    D = d * d

    # --- helpers --------------------------------------------------
    def metric(Wm):
        return np.eye(d) + Wm.T @ Wm

    def log_gauss(x, mean, chol):
        diff = x - mean
        y = np.linalg.solve(chol, diff)
        quad = 0.5 * y @ y
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        return -(quad + 0.5*D*np.log(2*np.pi) + 0.5*logdet)

    # --- current state -------------------------------------------
    W      = W_vec.copy()
    Wm     = W.reshape(d, d)
    grad   = gradlogpos_irrwriem_new(W, X, J, lambd)
    G      = metric(Wm)
    cholG  = cholesky(G)                         # G½  (lower-tri)

    mu_fwd = W + 0.5 * dt * grad
    eps    = np.random.randn(d, d)
    noise  = eps @ cholG
    W_prop = mu_fwd + np.sqrt(dt) * noise.ravel()

    # --- backward quantities -------------------------------------
    Wpm      = W_prop.reshape(d, d)
    cholG_p  = cholesky(metric(Wpm))
    grad_p   = gradlogpos_irrwriem_new(W_prop, X, J, lambd)
    mu_bwd   = W_prop + 0.5 * dt * grad_p

    log_q_fwd = log_gauss(W_prop, mu_fwd, np.sqrt(dt) * np.kron(cholG, np.eye(d)))
    log_q_bwd = log_gauss(W,      mu_bwd, np.sqrt(dt) * np.kron(cholG_p, np.eye(d)))

    log_alpha = (logpi(W_prop, X, lambd) - logpi(W, X, lambd)
                 + log_q_bwd - log_q_fwd)
    
    # Compute acceptance probability
    alpha = min(1.0, np.exp(log_alpha))
    
    # Generate random number for acceptance
    u = np.random.rand()
    accepted = u < alpha

    if accepted:
        return W_prop, True, alpha
    else:
        return W, False, alpha



def stat_func(Y):
    # Y: (d, K//N)
    # we compute some statistics of the chain
    return np.array([np.mean(np.sum(np.abs(Y), axis=0)), np.mean(np.sum(Y**2, axis=0)), \
                     np.mean(np.max(np.abs(Y), axis=0)), np.mean(np.max(Y, axis=0)), \
                     np.mean((Y[0, :] > 0) & (Y[1, :] > 0)), np.mean((Y[3, :] > 0) & (Y[4, :] > 0)), \
                     np.mean(np.sum(Y, axis=0)), np.mean(np.exp(np.sum(np.abs(Y), axis=0)/2))])


def main(n, dt, T, lambda_, num_chains, subsample_rate, path, initial_point=None):
    # Seed initialization
    np.random.seed(None)
    num_cpus = min(num_chains, os.cpu_count())
    N_stats = 8
    N = 100

    # Load ICA data globally (shared in memory)
    X = sio.loadmat('ica_data3.mat')['X']

    # Parameters
    d = X.shape[0]
    num_steps = int(T / dt)
    
    # Load initial point if provided
    if initial_point is not None:
        print(f'Loading initial point from {initial_point}')
        initial_data = np.load(initial_point, allow_pickle=True).item()
        if 'last_steps' in initial_data:
            # Use the last steps from previous run as initial points
            stored_last_steps = initial_data['last_steps']
            if len(stored_last_steps) >= num_chains:
                initial_points = stored_last_steps[:num_chains]
                print(f'Using {num_chains} stored last steps as initial points')
            else:
                print(f'Warning: Only {len(stored_last_steps)} stored points available, but {num_chains} chains requested')
                print('Using random initialization for remaining chains')
                initial_points = stored_last_steps.tolist()
                for i in range(num_chains - len(stored_last_steps)):
                    initial_points.append(np.diag(np.where(np.random.rand(d) > 0.5, 1, -1)).reshape(d**2))
        else:
            print('No last_steps found in initial point file, using random initialization')
            initial_points = None
    else:
        initial_points = None

    print('dt =', dt,
        'parameters: T =', T, 
        'num_steps =', num_steps,
        'num_chains =', num_chains, 
        'lambda_ =', lambda_, 'd =', d, 
        'subsample_rate =', subsample_rate
        )

    # Compute tensor product of J and I
    I = np.eye(3)
    J = np.kron(np.array([[0, 1, 1],
                        [-1, 0, 1], 
                        [-1, -1, 0]]), I) + \
        np.kron(I, np.array([[0, 1, 1],
                        [-1, 0, 1], 
                        [-1, -1, 0]]))

    print('Running chains...')
    
    # Print initialization method once
    if initial_points is not None:
        stored_count = len(initial_points)
        if stored_count >= num_chains:
            print(f'All {num_chains} chains will use stored initial points')
        else:
            print(f'{stored_count} chains will use stored initial points, {num_chains - stored_count} chains will use random initialization')
    else:
        print(f'All {num_chains} chains will use random initialization')
    
    def run_chain(ii, subsample_rate):
        first_entry = []
        stats = np.zeros((N, N_stats))
        
        # Initialize W based on whether initial points are provided
        if initial_points is not None and ii < len(initial_points):
            W = initial_points[ii].copy()
        else:
            W = np.diag(np.where(np.random.rand(d) > 0.5, 1, -1)).reshape(d**2)
        
        count = 0
        
        # Array to store acceptance probabilities
        alphas = []
        
        for nn in range(N):
            W_ir = np.zeros((d**2, num_steps//N+1))
            W_ir[:, 0] = W
            
            # Array for this block
            block_alphas = []
            
            for jj in range(num_steps//N):
                W, acc, alpha = irmala_step(W, X, J, lambda_, dt)
                W_ir[:, jj + 1] = W
                count += acc
                
                # Store acceptance probability
                block_alphas.append(alpha)
            
            # Store block data
            alphas.extend(block_alphas)
            
            W_ir_sub = W_ir[:,::subsample_rate]
            first_entry.append(W_ir_sub[0])
            stats[nn] = stat_func(W_ir_sub)
            W = W_ir[:, -1]

        # Store the last step (final state)
        last_step = W

        return stats, np.array(first_entry).flatten(), count/num_steps, np.array(alphas), last_step

    # Run chains in parallel
    print(f'cpu count: {os.cpu_count()}')
    print(f'num_chains: {num_chains}')
    print(f'num cpus: {num_cpus}')
    start_time = time()
    results = Parallel(n_jobs=num_cpus, backend='loky')(
        delayed(run_chain)(i, subsample_rate) for i in tqdm(range(num_chains))
    ) # N_chains x N x N_stats

    stats, first_entry, acc, alphas, last_steps = [np.array(x) for x in zip(*results)]
    end_time = time()
    print(f'Time taken: {end_time - start_time} seconds')

    # Print acceptance rate
    print(f'Average acceptance rate: {np.mean(acc):.4f}')

    # Unpack and average results
    print('stats.shape', np.array(stats).shape)
    print('first_entry.shape', np.array(first_entry).shape)
    print('alphas.shape', np.array(alphas).shape)
    print('last_steps.shape', np.array(last_steps).shape)
    
    # Save results
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{path}/main_giIrr_num_chains{num_chains}_dt{dt}_T{T}_lambda_{lambda_}_{n}.npy'
    np.save(filename, {
            'stats': stats, 
            'first_entry': first_entry,
            'acc': np.mean(acc),
            'alphas': alphas,
            'last_steps': last_steps
            })
    print('Statistics, acceptance probabilities, and last steps saved to', filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--dt', type=float, default=0.5*1e-4)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--lambda_', type=float, default=1) #precision (inverse variance) of the Gaussian prior on the weight matrix W
    parser.add_argument('--num_chains', type=int, default=25)
    parser.add_argument('--subsample_rate', type=int, default=500)
    parser.add_argument('--path', type=str, default='results/GiIrr')
    parser.add_argument('--initial_point', type=str, default=None, help='Path to .npy file containing initial point')
    args = parser.parse_args()
    main(args.n, args.dt, args.T, args.lambda_, args.num_chains, args.subsample_rate, args.path, args.initial_point)