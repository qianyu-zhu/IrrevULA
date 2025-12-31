import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Seed initialization
np.random.seed(None)

# Load data
ica_data = sio.loadmat('ica_data3.mat')
X_raw = ica_data['X']
X = X_raw[:2,::50]  # Adjust this according to your data structure
print('X.shape', X.shape)

def visualize_data(X_raw):
    # Plot each row of the original data
    plt.figure(figsize=(10, 6))
    for i in range(X_raw.shape[0]):
        plt.plot(X_raw[i,:], label=f'Component {i+1}')
        plt.title('ICA Data Components')
        plt.xlabel('Sample Index') 
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    # plt.show()
    plt.savefig('ica_data.png')


# Prior precision
lambda_ = 1
grid_level_list = [500]


def gradlogpos(W, X, lambda_):
    d, N = X.shape
    # n = int(0.1 * N)
    summat = np.zeros((d, d))
    W = W.reshape(d, d)

    index = np.random.choice(N, N, replace=False)
    for idx in index:
        xn = X[:, idx]
        yn = W @ xn
        summat += np.tanh(0.5 * yn)[:, np.newaxis] @ xn[np.newaxis, :]

    dlogpi = (N * np.linalg.inv(W.T) - (N / N) * summat) - lambda_ * W
    return dlogpi.flatten()

def density(W, X, lambda_):
    d, N = X.shape
    W = W.reshape(d, d)
    log_det_term = np.abs(np.linalg.det(W))
    # print(W[1,:] @ X)
    sum_p = np.sum(np.log(0.25 / np.cosh(0.5 * (W @ X))**2 ) )
    sum_N = -0.5 * lambda_ * np.sum(W ** 2)
    # print('sum_p, sum_N', sum_p, sum_N)
    log_density = log_det_term * N + sum_p + sum_N
    if np.isnan(np.exp(log_density)):
        print('log_density', log_density)
        print('W', W)
        print('lambda_', lambda_)
        print('log_det_term', log_det_term)
        print('sum_p', sum_p)
    # print('np.exp(log_density)', np.exp(log_density))
    return max(0, np.exp(log_density))




def expected_fisher_information(X, lambda_):
    W_samples = np.random.uniform(low=-2, hig=2, size=(10000, len(X), len(X)))
    weighted_sum = np.zeros((W_samples.shape[1], W_samples.shape[1]))
    total_density = 0
    
    for W in W_samples:
        grad = gradlogpos(W, X, lambda_)
        dens = density(W, X, lambda_)
        # print('grad.shape, dens.shape', grad.shape, dens.shape)
        weighted_sum += dens * np.outer(grad, grad)
        # print(dens)
        total_density += dens

    return weighted_sum / total_density

def visualize_density(density_func, lambda_, num_samples=10):
    dim = len(X)
    # Create grid points between -2 and 2 for each dimension
    grid_points = np.linspace(-2, 2, 20)
    
    # Create list to store samples
    samples = []
    
    # Iterate over grid points in each dimension
    for x1 in grid_points:
        for x2 in grid_points:
            for x3 in grid_points:
                for x4 in grid_points:
                    samples.append([x1, x2, x3, x4])
                    
    samples = np.array(samples)
    samples = samples.reshape(-1, dim, dim)
    
    # Calculate density at each grid point
    densities = np.array([density_func(W, X, lambda_) for W in samples])
    # Normalize densities to avoid numerical issues 
    densities = densities / np.sum(densities)

    fig, axes = plt.subplots(dim**2, dim**2, figsize=(10, 10))
    # plt.subplots_adjust(hspace=0.4, wspace=0.4)

    samples = samples.reshape(len(samples), dim**2)
    for i in range(dim**2):
        for j in range(dim**2):
            ax = axes[i, j]
            if i == j:
                # Plot KDE of the i-th dimension
                from scipy.stats import gaussian_kde

                x_vals = samples[:, i]
                kde = gaussian_kde(x_vals, weights=densities)
                xs = np.linspace(-2, 2, 200)
                ax.plot(xs, kde(xs), color='black')
                ax.set_yticks([])
                ax.set_xticks([])
            else:
                xi = samples[:, j]
                xj = samples[:, i]
                pts_2d = np.vstack((xi, xj)).T
                grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:100j]
                grid_z = griddata(pts_2d, densities, (grid_x, grid_y), method='linear')

                ax.contourf(grid_x, grid_y, grid_z, levels=30, cmap='viridis')
                ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"Full {dim**2}x{dim**2} Density Projection Grid", fontsize=24)
    plt.tight_layout()
    plt.show()
    return



visualize_density(density, lambda_, num_samples=5000)

exit()

# compute expected fisher information matrix
M_list = [10000, 50000, 100000, 150000, 200000, 300000, 400000, 500000]
for M in tqdm(M_list):
    samples = map_samples(M)
    weights = np.zeros((M))
    fisher_info = np.zeros((4, 4))
    for i in range(M):
        mu1, sigma1, mu2, sigma2 = samples[i, 0], samples[i, 1], samples[i, 2], samples[i, 3]
        weights[i] = 1/sigma1**N * np.exp( np.sum( -(X1 - mu1)**2 / (2 * sigma1**2) ) ) * 1/sigma2**N * np.exp( np.sum( -(X2 - mu2)**2 / (2 * sigma2**2) ) )
        fisher_info += get_fisher(mu1, sigma1, mu2, sigma2, N) * weights[i]
    fisher_info = fisher_info / np.sum(weights) # normalize

    print(np.round(fisher_info, decimals=2))
# Save fisher information
np.save(f'statistics/fisher_information_{M}.npy', fisher_info)


"""
# compute first and second moments
Emu_list, Esigma_list, Emu2_list, Esigma2_list = [], [], [], []
print('--------------X1------------------')
for grid_level in tqdm(grid_level_list):
    weights = np.zeros((grid_level+1, grid_level+1))
    Emu_ = np.zeros((grid_level+1, grid_level+1))
    Esigma_ = np.zeros((grid_level+1, grid_level+1))
    Emu2_ = np.zeros((grid_level+1, grid_level+1))
    Esigma2_ = np.zeros((grid_level+1, grid_level+1))
    mu_grid = np.linspace(-1.5, 1, grid_level+1)
    sigma_grid = np.linspace(1, 3.5, grid_level+1)

    for i in range(grid_level+1):
        mu = mu_grid[i]
        for j in range(grid_level+1):
            sigma = sigma_grid[j]
            weights[i, j] = 1/sigma**N * np.exp( np.sum( -(X1 - mu)**2 / (2 * sigma**2) ) )
            Emu_[i, j] = weights[i, j] * mu_grid[i]
            Esigma_[i, j] = weights[i, j] * sigma_grid[j]
            Emu2_[i, j] = weights[i, j] * mu_grid[i]**2
            Esigma2_[i, j] = weights[i, j] * sigma_grid[j]**2
            
    Emu_list.append(np.sum(Emu_)/np.sum(weights))
    Esigma_list.append(np.sum(Esigma_)/np.sum(weights))
    Emu2_list.append(np.sum(Emu2_)/np.sum(weights))
    Esigma2_list.append(np.sum(Esigma2_)/np.sum(weights))

    
    plt.figure(figsize=(8, 6))
    plt.imshow(weights)
    y_ticks = [grid_level//5*i for i in range(6)]
    y_labels = [f'{mu_grid[i*grid_level//5]:.2f}' for i in range(6)]
    x_ticks = [grid_level//5*i for i in range(6)]
    x_labels = [f'{sigma_grid[i*grid_level//5]:.2f}' for i in range(6)]

    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.ylabel(f'mu: [{mu_grid[0]}, {mu_grid[-1]}]')
    plt.xlabel(f'sigma: [{sigma_grid[0]}, {sigma_grid[-1]}]')
    plt.colorbar()
    plt.title(f'mu: {mu_true[0]}, sigma: {sigma_true[0]}, grid_level: {grid_level}')
    plt.savefig(f'mu: {mu_true[0]}, sigma: {sigma_true[0]}, grid_level: {grid_level}.png')
    plt.close()
    # plt.show()

# sys.exit()
print('Grid Level Results for X1:')
print('--------------------------------')
for i, grid_level in enumerate(grid_level_list):
    print(f'Grid Level: {grid_level}')
    print(f'E[μ]: {Emu_list[i]:.15f}')
    print(f'E[σ]: {Esigma_list[i]:.15f}') 
    print(f'E[μ²]: {Emu2_list[i]:.15f}')
    print(f'E[σ²]: {Esigma2_list[i]:.15f}')
    print('--------------------------------')



Emu_list, Esigma_list, Emu2_list, Esigma2_list = [], [], [], []
print('--------------X2------------------')
for grid_level in tqdm(grid_level_list):
    weights = np.zeros((grid_level+1, grid_level+1))
    Emu_ = np.zeros((grid_level+1, grid_level+1))
    Esigma_ = np.zeros((grid_level+1, grid_level+1))
    Emu2_ = np.zeros((grid_level+1, grid_level+1))
    Esigma2_ = np.zeros((grid_level+1, grid_level+1))
    mu_grid = np.linspace(-14, 12, grid_level+1)
    sigma_grid = np.linspace(12, 38, grid_level+1)

    for i in range(grid_level+1):
        for j in range(grid_level+1):
            weights[i, j] = 1/sigma_grid[j]**N * np.exp( np.sum( -(X2 - mu_grid[i])**2 / (2 * sigma_grid[j]**2) ) )
            Emu_[i, j] = weights[i, j] * mu_grid[i]
            Esigma_[i, j] = weights[i, j] * sigma_grid[j]
            Emu2_[i, j] = weights[i, j] * mu_grid[i]**2
            Esigma2_[i, j] = weights[i, j] * sigma_grid[j]**2
            
    Emu_list.append(np.sum(Emu_)/np.sum(weights))
    Esigma_list.append(np.sum(Esigma_)/np.sum(weights))
    Emu2_list.append(np.sum(Emu2_)/np.sum(weights))
    Esigma2_list.append(np.sum(Esigma2_)/np.sum(weights))

    plt.figure(figsize=(8, 6))
    plt.imshow(weights)
    y_ticks = [grid_level//5*i for i in range(6)]
    y_labels = [f'{mu_grid[i*grid_level//5]:.2f}' for i in range(6)]
    x_ticks = [grid_level//5*i for i in range(6)]
    x_labels = [f'{sigma_grid[i*grid_level//5]:.2f}' for i in range(6)]

    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.ylabel(f'mu: [{mu_grid[0]}, {mu_grid[-1]}]')
    plt.xlabel(f'sigma: [{sigma_grid[0]}, {sigma_grid[-1]}]')
    plt.colorbar()
    plt.title(f'mu: {mu_true[1]}, sigma: {sigma_true[1]}, grid_level: {grid_level}')
    plt.savefig(f'mu: {mu_true[1]}, sigma: {sigma_true[1]}, grid_level: {grid_level}.png')
    plt.close()
    # plt.show()

print('Grid Level Results for X2:')
print('--------------------------------')
for i, grid_level in enumerate(grid_level_list):
    print(f'Grid Level: {grid_level}')
    print(f'E[μ]: {Emu_list[i]:.15f}')
    print(f'E[σ]: {Esigma_list[i]:.15f}') 
    print(f'E[μ²]: {Emu2_list[i]:.15f}')
    print(f'E[σ²]: {Esigma2_list[i]:.15f}')
    print('--------------------------------')


# Save statistics to files
np.save('statistics/Emu_list.npy', np.array(Emu_list))
np.save('statistics/Esigma_list.npy', np.array(Esigma_list))
np.save('statistics/Emu2_list.npy', np.array(Emu2_list))
np.save('statistics/Esigma2_list.npy', np.array(Esigma2_list))

# Save grid levels for reference
np.save('statistics/grid_levels.npy', np.array(grid_level_list))
"""



"""
# visualize the posterior distribution
grid_level = 500
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
for idx_i in range(4):
    for idx_j in range(4):
        weights = np.zeros((grid_level+1, grid_level+1))
        mu_grid = np.linspace(-15, 15, grid_level+1)
        sigma_grid = np.linspace(0.1, 30, grid_level+1)

        for i in range(grid_level+1):
            mu = mu_grid[i]
            for j in range(grid_level+1):
                sigma = sigma_grid[j]
                weights[i, j] = 1/sigma**N * np.exp( np.sum( -(X2 - mu)**2 / (2 * sigma**2) ) )
        axs[idx_i, idx_j].imshow(weights)
        y_ticks = [grid_level//5*i for i in range(6)]
        y_labels = [f'{mu_grid[i*grid_level//5]:.2f}' for i in range(6)]
        x_ticks = [grid_level//5*i for i in range(6)]
        x_labels = [f'{sigma_grid[i*grid_level//5]:.2f}' for i in range(6)]

        axs[idx_i, idx_j].xticks(x_ticks, x_labels)
        axs[idx_i, idx_j].yticks(y_ticks, y_labels)
        axs[idx_i, idx_j].ylabel(f'mu: [{mu_grid[0]}, {mu_grid[-1]}]')
        axs[idx_i, idx_j].xlabel(f'sigma: [{sigma_grid[0]}, {sigma_grid[-1]}]')
        axs[idx_i, idx_j].colorbar()
        axs[idx_i, idx_j].title(f'mu: {mu_true[1]}, sigma: {sigma_true[1]}, grid_level: {grid_level}')

plt.savefig(f'mu: {mu_true[1]}, sigma: {sigma_true[1]}, grid_level: {grid_level}.png')
plt.close()
"""