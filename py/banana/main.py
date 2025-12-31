import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from getoptJ import getJ, getnoptJ, getoptJ


# Parallel processing using multiprocessing
from multiprocessing import Pool
import multiprocessing as mp

# --- Banana log density and gradient ---
def log_banana_density(x, b=0.1):
    """
    Log density of banana-shaped distribution.
    """
    y1 = x[0]
    y2 = x[1] + b * x[0] ** 2
    return -0.5 * (y1 ** 2 + y2 ** 2)

def grad_log_banana_density(x, b=0.1):
    """
    Gradient of log banana density.
    """
    y1 = x[0]
    y2 = x[1] + b * x[0] ** 2

    dy1_dx0 = 1
    dy2_dx0 = 2 * b * x[0]
    dy2_dx1 = 1

    dlogp_dx0 = -(y1 * dy1_dx0 + y2 * dy2_dx0)
    dlogp_dx1 = -(y2 * dy2_dx1)

    return np.array([dlogp_dx0, dlogp_dx1])



# --- Langevin MCMC (MALA) ---
def main(M, h, T, b, path):
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Set random seed
    np.random.seed(42)

    d = 2
    # Discretization
    K = int(T / h) + 1 # number of steps
    initcond = np.array([5, 5]) # initial condition, true value is [-0.7, 4.4, -1.6, 10.3]
    x = initcond

    firstmoment = np.sum(np.array([0, -b]))
    secondmoment = np.sum(np.array([1, 1+2*b**2]))

    fisher = np.load(f'fisher_stats/banana_fisher_{b}.npy')
    inv_fisher = np.linalg.inv(fisher)
    print('fisher', np.round(fisher, decimals=3))
    J1, J2, J3 = [getJ(fisher) for _ in range(3)]
    J_nopt1, J_nopt2, J_nopt3 = [getnoptJ(fisher) for _ in range(3)]
    J_opt1, J_opt2, J_opt3 = [getoptJ(fisher) for _ in range(3)]


    print('spectrum of random J')
    for J in [J1, J2, J3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
        print('(J@fisher@J)', np.round(np.trace(J@fisher@J), decimals=3))
        print('(J@fisher@J@fisher)', np.round(np.trace(J@fisher@J@fisher), decimals=3))
    print('spectrum of nopt J')
    for J in [J_nopt1, J_nopt2, J_nopt3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
        print('(J@fisher@J)', np.round(np.trace(J@fisher@J), decimals=3))
        print('(J@fisher@J@fisher)', np.round(np.trace(J@fisher@J@fisher), decimals=3))
    print('spectrum of opt J')
    for J in [J_opt1, J_opt2, J_opt3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
        print('(J@fisher@J)', np.round(np.trace(J@fisher@J), decimals=3))
        print('(J@fisher@J@fisher)', np.round(np.trace(J@fisher@J@fisher), decimals=3))
    # Initialize arrays
    Yall = np.zeros((d, K, M)) # dimension * stepsize * chain
    Yrm2all = np.zeros((d, K, M))

    Yir1all, Yir2all, Yir3all = [np.zeros((d, K, M)) for _ in range(3)]
    Ynopt1all, Ynopt2all, Ynopt3all = [np.zeros((d, K, M)) for _ in range(3)]
    Yopt1all, Yopt2all, Yopt3all = [np.zeros((d, K, M)) for _ in range(3)]

    gradfunc = grad_log_banana_density

    # Initialize arrays
    Yall = np.zeros((d, K, M)) # dimension * stepsize * chain
    Yrm2all = np.zeros((d, K, M))

    Yir1all, Yir2all, Yir3all = [np.zeros((d, K, M)) for _ in range(3)]
    Ynopt1all, Ynopt2all, Ynopt3all = [np.zeros((d, K, M)) for _ in range(3)]
    Yopt1all, Yopt2all, Yopt3all = [np.zeros((d, K, M)) for _ in range(3)]

    # Start timing
    start_time = time.time()

    # Langevin sampling
    for mm in tqdm(range(M)):
        # Standard Langevin
        Y = np.zeros((d, K))
        Y[:, 0] = initcond
        for kk in range(K - 1):
            Y[:, kk + 1] = Y[:, kk] + h / 2 * gradfunc(Y[:, kk], b=b) + np.sqrt(h) * np.random.randn(d)
        Yall[:, :, mm] = Y

        # RM Langevin constant
        Yrm2 = np.zeros((d, K))
        Yrm2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yrm2[:, kk], b=b)
            Yrm2[:, kk + 1] = Yrm2[:, kk] + h / 2 * (inv_fisher @ gradeval) + sqrtm(h * inv_fisher) @ np.random.randn(d)
        Yrm2all[:, :, mm] = Yrm2

        # Irreversible Langevin
        Yir1 = np.zeros((d, K))
        Yir1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir1[:, kk], b=b)
            Yir1[:, kk + 1] = Yir1[:, kk] + h / 2 * (np.eye(d) + J1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir1all[:, :, mm] = Yir1

        Yir2 = np.zeros((d, K))
        Yir2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir2[:, kk], b=b)
            Yir2[:, kk + 1] = Yir2[:, kk] + h / 2 * (np.eye(d) + J2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir2all[:, :, mm] = Yir2

        Yir3 = np.zeros((d, K))
        Yir3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir3[:, kk], b=b)
            Yir3[:, kk + 1] = Yir3[:, kk] + h / 2 * (np.eye(d) + J3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir3all[:, :, mm] = Yir3

        Ynopt1 = np.zeros((d, K))
        Ynopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt1[:, kk], b=b)
            Ynopt1[:, kk + 1] = Ynopt1[:, kk] + h / 2 * (np.eye(d) + J_nopt1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt1all[:, :, mm] = Ynopt1

        Ynopt2 = np.zeros((d, K))
        Ynopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt2[:, kk], b=b)
            Ynopt2[:, kk + 1] = Ynopt2[:, kk] + h / 2 * (np.eye(d) + J_nopt2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt2all[:, :, mm] = Ynopt2

        Ynopt3 = np.zeros((d, K))
        Ynopt3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt3[:, kk], b=b)
            Ynopt3[:, kk + 1] = Ynopt3[:, kk] + h / 2 * (np.eye(d) + J_nopt3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt3all[:, :, mm] = Ynopt3

        # Optimal Langevin
        Yopt1 = np.zeros((d, K))
        Yopt1[:, 0] = initcond
        for kk in range(K - 1):
            Yopt1[:, kk + 1] = Yopt1[:, kk] + h / 2 * (np.eye(d) + J_opt1) @ gradfunc(Yopt1[:, kk], b=b) + np.sqrt(h) * np.random.randn(d)
        Yopt1all[:, :, mm] = Yopt1

        Yopt2 = np.zeros((d, K))
        Yopt2[:, 0] = initcond
        for kk in range(K - 1):
            Yopt2[:, kk + 1] = Yopt2[:, kk] + h / 2 * (np.eye(d) + J_opt2) @ gradfunc(Yopt2[:, kk], b=b) + np.sqrt(h) * np.random.randn(d)
        Yopt2all[:, :, mm] = Yopt2

        Yopt3 = np.zeros((d, K))
        Yopt3[:, 0] = initcond
        for kk in range(K - 1):
            Yopt3[:, kk + 1] = Yopt3[:, kk] + h / 2 * (np.eye(d) + J_opt3) @ gradfunc(Yopt3[:, kk], b=b) + np.sqrt(h) * np.random.randn(d)
        Yopt3all[:, :, mm] = Yopt3
    # End timing and store execution time

    end_time = time.time()
    running_time = end_time - start_time

    # burnin = int(50 / h)
    burnin = 0

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)

    Y_list = [Yall, Yrm2all, Yir1all, Yir2all, Yir3all, Ynopt1all, Ynopt2all, Ynopt3all, Yopt1all, Yopt2all, Yopt3all]
    methods = ['Standard', 'RM_const', 'irr_1', 'irr_2', 'irr_3', 'nopt_1', 'nopt_2', 'nopt_3', 'opt_1', 'opt_2', 'opt_3']
    colors = ['black', '#1a80bb', '#a00000', '#a00000', '#a00000', '#ea801c', '#ea801c', '#ea801c', '#f2c45f', '#f2c45f', '#f2c45f']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # Initialize estimators
    # Calculate number of steps with step_size
    base = 1.5
    length = int(np.floor(np.log(K - burnin) / np.log(base)))
    x_list = [int(base ** i) for i in range(1, length+1)] + [K-burnin]

    estimatorfirst = np.zeros((len(Y_list), len(x_list), M))
    estimatorsec = np.zeros((len(Y_list), len(x_list), M)) 
    start_time = time.time()

    for idx, Y in enumerate(Y_list):
        Y_slice = Y[:, burnin:, :]
        for i in range(len(x_list)):
            estimatorfirst[idx, i, :] = np.mean(Y_slice.sum(axis=0)[:x_list[i], :], axis=0)
            estimatorsec[idx, i, :] = np.mean((Y_slice ** 2).sum(axis=0)[:x_list[i], :], axis=0)
        print(np.mean(estimatorfirst[idx,-1,:]))
        print(np.mean(estimatorsec[idx,-1,:]))
    # print(estimatorfirst)

    # End timing and store execution time
    end_time = time.time()
    estimating_time = end_time - start_time


    # Start timing
    start_time = time.time()
    # Compute Mean Squared Errors and Biases
    mse_first = np.zeros((len(Y_list), length+1))
    bias_first_sq = np.zeros((len(Y_list), length+1))
    for idx in range(len(Y_list)):
        mse = np.mean((estimatorfirst[idx] - firstmoment) ** 2, axis=1)
        biassq = np.mean(estimatorfirst[idx] - firstmoment, axis=1) ** 2
        mse_first[idx] = mse
        bias_first_sq[idx] = biassq

    # Second moment MSE calculations commented out
    mse_sec = np.zeros((len(Y_list), length+1))
    bias_sec_sq = np.zeros((len(Y_list), length+1))
    for idx in range(len(Y_list)):
        mse = np.mean((estimatorsec[idx] - secondmoment) ** 2, axis=1)
        biassq = np.mean(estimatorsec[idx] - secondmoment, axis=1) ** 2
        mse_sec[idx] = mse
        bias_sec_sq[idx] = biassq

    # Save statistics
    # np.save(f'{path}/x_list_T{T}_M{M}_h{h}.npy', x_list)
    # np.save(f'{path}/mse_first_T{T}_M{M}_h{h}.npy', mse_first)
    # np.save(f'{path}/bias_first_sq_T{T}_M{M}_h{h}.npy', bias_first_sq)
    # np.save(f'{path}/mse_sec_T{T}_M{M}_h{h}.npy', mse_sec)
    # np.save(f'{path}/bias_sec_sq_T{T}_M{M}_h{h}.npy', bias_sec_sq)


    # End timing and store execution time
    end_time = time.time()
    computing_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")
    print(f"Computing Estimators: {estimating_time:.2f} seconds")
    print(f"Computing MSE and Bias: {computing_time:.2f} seconds")

    # Plotting

    # Plot MSE and bias
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    plot_data = [
        (mse_first, 'First Moment MSE', 'MSE'),
        (mse_sec, 'Second Moment MSE', 'MSE'),
        (bias_first_sq, 'First Moment Squared Bias', 'Squared Bias'),
        (bias_sec_sq, 'Second Moment Squared Bias', 'Squared Bias')
    ]
    
    for (data, title, ylabel), ax in zip(plot_data, axes1.flat):
        for idx in range(len(Y_list)):
            ax.loglog(data[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
        ax.set_xlabel('steps')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', ls='-', alpha=0.2)
        ax.grid(True, which='minor', ls=':', alpha=0.2)

    plt.tight_layout()
    plt.title(f'Banana with b={b}')
    plt.savefig(f'{path}/mse_bias_Banana_b{b}_T{T}_M{M}_h{h}.png')

    # Plot trajectories
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 12))
    grid_level = 101
    
    def plot_density_and_trajectories(ax, x_range, y_range, density_func):
        x = np.linspace(*x_range, grid_level)
        y = np.linspace(*y_range, grid_level) 
        X_grid, Y_grid = np.meshgrid(x, y)
        Z = density_func(X_grid, Y_grid, b)
        
        contour = ax.contour(x, y, Z, levels=10)
        plt.colorbar(contour, ax=ax)
        ax.set_xlabel('mu')
        ax.set_ylabel('sigma')
        
        return contour

    # First density plot
    def banana_density(X_grid, Y_grid, b):
        Z = np.zeros_like(X_grid)
        for i in range(len(X_grid)):
            for j in range(len(Y_grid)):
                Z[j,i] = np.exp(log_banana_density([X_grid[i,j], Y_grid[i,j]], b=b))
        return Z


    # Plot trajectories
    for idx, trajectory in enumerate(Y_list):
        axes2[0].plot(trajectory[0,:4000,0], trajectory[1,:4000,0], 
                       label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
        axes2[1].plot(np.mean(trajectory[0,:4000,:], axis=1), np.mean(trajectory[1,:4000,:], axis=1),
                       label=methods[idx], linewidth=1, color=colors[idx])
    axes2[0].legend()
    axes2[1].legend()



    plt.suptitle(f'trajectory plot on Banana {b}, different J, T={T}, M={M}, h={h}')
    plt.savefig(f'{path}/trajectory_Banana_b{b}_T{T}_M{M}_h{h}.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.01)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--b', type=float, default=0.5)
    parser.add_argument('--path', type=str, default='banana')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, b=args.b, path=args.path)