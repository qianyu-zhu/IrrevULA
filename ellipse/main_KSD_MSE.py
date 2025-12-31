import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ, getJ


"""
Simplified MCMC comparison script with 6 methods:
1. Unperturbed (vanilla Langevin)
2. rand-S (small perturbation)
3. rand-M (medium perturbation)
4. rand-L (large perturbation)
5. spec (spectral-optimal perturbation)
6. spec-E (spectral-optimal & ESJD^W-optimal perturbation)
"""

def compute_statistics(Y):
    """Compute 4 statistics for a chunk of samples"""
    # Y: (d, chunk_size)
    return np.mean(np.array([
        np.sum(np.abs(Y), axis=0),           # Sum of absolute values
        np.sum(Y**2, axis=0),                # Sum of squared values
        np.max(np.abs(Y), axis=0),           # Maximum absolute value
        (np.max(Y,axis=0) > 16).astype(int)          # Binary threshold indicator
    ]), axis=1)


def simulate_single_chain(seed, step_size, total_time, sigma_true, subsample_rate=10):
    """Simulate one MCMC chain with all 6 methods"""
    # Setup
    d = 4
    num_chunks = 20
    num_stats = 4
    total_steps = int(total_time / step_size)
    subsampled_steps = total_steps // subsample_rate
    
    np.random.seed(seed)
    mu_true = np.zeros(d)
    cov_true = np.diag(np.array(sigma_true)**2)
    fisher = np.linalg.inv(cov_true)
    
    # Average norm for J scaling
    opt_norm_ave = 9.13
    
    # Method configurations
    methods = [
        ('unperturbed', np.zeros((d, d))),                    # No perturbation
        ('irr-S', getJ(np.diag(sigma_true))* opt_norm_ave / 2),    # Small perturbation
        ('irr-M', getJ(np.diag(sigma_true))* opt_norm_ave),             # Medium perturbation
        ('irr-L', getJ(np.diag(sigma_true))* opt_norm_ave * 1.5),    # Large perturbation
        ('irr-Spec', getnoptJ(fisher)),   # Sub-optimal perturbation
        ('irr-SE', getoptJ(fisher))               # Optimal perturbation
    ]
    print('methods', methods)
    
    results = {}
    
    for method_name, J_matrix in methods:
        # Initialize chain
        Y = np.random.multivariate_normal(mu_true, cov_true)
        Y_chain = np.zeros((d, total_steps))
        Y_chain[:, 0] = Y
        
        # Simulate full chain
        for step in range(total_steps - 1):
            grad = gradfunc(Y_chain[:, step], fisher)
            drift = (np.eye(d) + J_matrix) @ grad
            
            # Langevin step
            Y_chain[:, step + 1] = (Y_chain[:, step] + 
                                   step_size / 2 * drift + 
                                   np.sqrt(step_size) * np.random.randn(d))
        
        # Subsample and compute statistics
        Y_subsampled = Y_chain[:, ::subsample_rate]
        chunk_size = subsampled_steps // num_chunks
        
        stats = np.zeros((num_chunks, num_stats))
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size if chunk_idx < num_chunks - 1 else subsampled_steps
            Y_chunk = Y_subsampled[:, start_idx:end_idx]
            stats[chunk_idx] = compute_statistics(Y_chunk)
        
        results[method_name] = {
            'full_chain': Y_chain,
            'statistics': stats
        }
    return results

def main(M, h, T, sigma_true, path, subsample_rate=10):
    """Main function"""
    print(f"Running MCMC comparison with parameters:")
    print(f"  Number of chains: {M}")
    print(f"  Step size: {h}")
    print(f"  Total time: {T}")
    print(f"  Sigma: {sigma_true}")
    print(f"  Subsample rate: {subsample_rate}")
    
    # Create output directory
    if path and not os.path.exists(path):
        os.makedirs(path)
    
    # Run simulations
    print(f"\nRunning {M} chains in parallel...")
    start_time = time.time()
    
    num_cores = min(multiprocessing.cpu_count(), 16)
    with Parallel(n_jobs=num_cores, backend="multiprocessing") as parallel:
        results = parallel(delayed(simulate_single_chain)(
            seed, h, T, sigma_true, subsample_rate) for seed in tqdm(range(M)))
    
    running_time = time.time() - start_time
    print(f"Simulation completed in {running_time:.2f} seconds")
    
    # Combine results from all chains
    method_names = ['unperturbed', 'rand-S', 'rand-M', 'rand-L', 'spec', 'spec-E']
    combined_results = {method: {'statistics': [], 'full_chain': []} for method in method_names}
    
    for chain_result in results:
        for method_name in method_names:
            combined_results[method_name]['statistics'].append(chain_result[method_name]['statistics'])
            combined_results[method_name]['full_chain'].append(chain_result[method_name]['full_chain'])
    
    # Stack results
    for method_name in method_names:
        combined_results[method_name]['statistics'] = np.array(combined_results[method_name]['statistics'])
        combined_results[method_name]['full_chain'] = np.array(combined_results[method_name]['full_chain'])
    
    # Calculate trajectory length for asymptotic variance
    total_steps = int(T / h)
    subsampled_steps = total_steps // subsample_rate
    trajectory_length = subsampled_steps
    
    print(f"\nTrajectory parameters:")
    print(f"  Total steps: {total_steps}")
    print(f"  Subsampled steps: {subsampled_steps}")
    print(f"  Trajectory length: {trajectory_length}")
    
    
    np.save(f'{path}/results_T{T}_M{M}_h{h}.npy', combined_results)
    print(f"\nResults saved to {path}/results_T{T}_M{M}_h{h}.npy")
    print(f"Note: Only analysis results saved (no full chains stored)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCMC Method Comparison')
    parser.add_argument('--M', type=int, default=20, help='Number of MCMC chains')
    parser.add_argument('--h', type=float, default=0.04, help='Step size')
    parser.add_argument('--T', type=int, default=10000, help='Total simulation time')
    parser.add_argument('--sigma_true', type=float, nargs='+', default=[2**i for i in range(4)], help='True standard deviations')
    parser.add_argument('--path', type=str, default='results', help='Output directory')
    parser.add_argument('--subsample_rate', type=int, default=50, help='Subsampling rate')
    
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, sigma_true=args.sigma_true, path=args.path, subsample_rate=args.subsample_rate)
