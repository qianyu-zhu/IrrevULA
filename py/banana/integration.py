import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def log_banana_density(x, b=0.2, c=0.00, sigma1=1.0, sigma2=0.3):
    y1 = x[0] / sigma1
    y2 = (x[1] + b * x[0] ** 2 + c * x[0] ** 3) / sigma2
    return -0.5 * (y1 ** 2 + y2 ** 2)

def banana_density(x, b=0.2, c=0.00, sigma1=1.0, sigma2=0.3):
    return np.exp(log_banana_density(x, b=b, c=c, sigma1=sigma1, sigma2=sigma2))

# --- Gradient of log-density ---
def grad_log_banana_density(x, b=0.2, c=0.00, sigma1=1.0, sigma2=0.3):
    x1, x2 = x
    # intermediate variables
    y1 = x1 / sigma1
    y2 = (x2 + b * x1**2 + c * x1**3) / sigma2
    
    # gradients
    dy1_dx1 = 1 / sigma1
    dy2_dx1 = (2 * b * x1 + 3 * c * x1**2) / sigma2
    dy2_dx2 = 1 / sigma2
    
    # d log pi / dx1
    dlogpi_dx1 = -(y1 * dy1_dx1 + y2 * dy2_dx1)
    # d log pi / dx2
    dlogpi_dx2 = -y2 * dy2_dx2

    return np.array([dlogpi_dx1, dlogpi_dx2])


# --- Plot the banana density ---
def plot_density_with_trajectory(samples, b=0.2, c=0.00, sigma1=1.0, sigma2=0.3, xlim=(-4, 4), ylim=(-4, 4), grid_size=200):
    """
    Plot banana density and overlay MCMC trajectory as a line.
    """
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    fisher = np.zeros((2,2))

    # Evaluate banana density on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.exp(log_banana_density([X[i, j], Y[i, j]], b=b, c=c, sigma1=sigma1, sigma2=sigma2))
            grad = grad_log_banana_density([X[i, j], Y[i, j]], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            fisher += Z[i, j] * np.outer(grad, grad)
    fisher = fisher / np.sum(Z)
    # print('fisher', np.round(fisher, decimals=3))
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Density')

    # Plot MCMC trajectory
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], color='red', lw=1, alpha=0.8, label='MCMC trajectory')
        plt.scatter(samples[::200, 0], samples[::200, 1], color='white', s=10, label='Trajectory points', zorder=5)

        # Optionally mark start and end
        plt.scatter(samples[0, 0], samples[0, 1], color='green', s=60, label='Start', marker='o', zorder=6)
        plt.scatter(samples[-1, 0], samples[-1, 1], color='black', s=60, label='End', marker='x', zorder=6)

    plt.title(f"Banana Density with MCMC Trajectory (b={b}, c={c}, sigma1={sigma1}, sigma2={sigma2})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'density_complex/banana_density_b_{b}_c_{c}_sigma1_{sigma1}_sigma2_{sigma2}.png')
    np.save(f'fisher_stats_complex/banana_fisher_{b}_c_{c}_sigma1_{sigma1}_sigma2_{sigma2}.npy', fisher)
    # plt.show()
    return fisher

# b = 0.3
# c = 0.0
# sigma1 = 1.0
# sigma2 = 0.3
# plot_density_with_trajectory(None, b=b, c=c, sigma1=sigma1, sigma2=sigma2, xlim=(-4, 4), ylim=(-4, 4), grid_size=200)