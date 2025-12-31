import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ, getJ

def main(M,h,T,path):
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Set random seed
    np.random.seed(42)

    # Data
    N = 30
    mu_true = [0, 0]
    sigma_true = [1, 5]
    X = mu_true + sigma_true * np.random.randn(N, 2) # random initialization
    d = X.shape[1]*2

    # Discretization
    K = int(T / h) + 1 # number of steps
    initcond = np.array([1, 1, 1, 1]) # initial condition, true value is ~[-0.7, 4.4, -1.6, 10.3]

    # Functions
    B = lambda state: (1 / N) * np.diag([state[1] ** 2, state[1] ** 2 / 2, state[3] ** 2, state[3] ** 2 / 2]) # inverse Fisher

    fisher = np.load('statistics/fisher_information_2_100000.npy')
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

    # Start timing
    start_time = time.time()
    # Langevin sampling
    for mm in tqdm(range(M)):
        # Standard Langevin
        Y = np.zeros((d, K))
        Y[:, 0] = initcond
        for kk in range(K - 1):
            Y[:, kk + 1] = Y[:, kk] + h / 2 * gradfunc(Y[:, kk], X) + np.sqrt(h) * np.random.randn(d)
        Yall[:, :, mm] = Y


        # RM Langevin constant
        Yrm2 = np.zeros((d, K))
        Yrm2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yrm2[:, kk], X)
            Yrm2[:, kk + 1] = Yrm2[:, kk] + h / 2 * (inv_fisher @ gradeval) + sqrtm(h * inv_fisher) @ np.random.randn(d)
        Yrm2all[:, :, mm] = Yrm2

        # Irreversible Langevin
        Yir1 = np.zeros((d, K))
        Yir1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir1[:, kk], X)
            Yir1[:, kk + 1] = Yir1[:, kk] + h / 2 * (np.eye(d) + J1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir1all[:, :, mm] = Yir1

        Yir2 = np.zeros((d, K))
        Yir2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir2[:, kk], X)
            Yir2[:, kk + 1] = Yir2[:, kk] + h / 2 * (np.eye(d) + J2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir2all[:, :, mm] = Yir2

        Yir3 = np.zeros((d, K))
        Yir3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir3[:, kk], X)
            Yir3[:, kk + 1] = Yir3[:, kk] + h / 2 * (np.eye(d) + J3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir3all[:, :, mm] = Yir3

        Ynopt1 = np.zeros((d, K))
        Ynopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt1[:, kk], X)
            Ynopt1[:, kk + 1] = Ynopt1[:, kk] + h / 2 * (np.eye(d) + J_nopt1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt1all[:, :, mm] = Ynopt1

        Ynopt2 = np.zeros((d, K))
        Ynopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt2[:, kk], X)
            Ynopt2[:, kk + 1] = Ynopt2[:, kk] + h / 2 * (np.eye(d) + J_nopt2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt2all[:, :, mm] = Ynopt2

        Ynopt3 = np.zeros((d, K))
        Ynopt3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt3[:, kk], X)
            Ynopt3[:, kk + 1] = Ynopt3[:, kk] + h / 2 * (np.eye(d) + J_nopt3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt3all[:, :, mm] = Ynopt3

        # Optimal Langevin
        Yopt1 = np.zeros((d, K))
        Yopt1[:, 0] = initcond
        for kk in range(K - 1):
            Yopt1[:, kk + 1] = Yopt1[:, kk] + h / 2 * (np.eye(d) + J_opt1) @ gradfunc(Yopt1[:, kk], X) + np.sqrt(h) * np.random.randn(d)
        Yopt1all[:, :, mm] = Yopt1

        Yopt2 = np.zeros((d, K))
        Yopt2[:, 0] = initcond
        for kk in range(K - 1):
            Yopt2[:, kk + 1] = Yopt2[:, kk] + h / 2 * (np.eye(d) + J_opt2) @ gradfunc(Yopt2[:, kk], X) + np.sqrt(h) * np.random.randn(d)
        Yopt2all[:, :, mm] = Yopt2

        Yopt3 = np.zeros((d, K))
        Yopt3[:, 0] = initcond
        for kk in range(K - 1):
            Yopt3[:, kk + 1] = Yopt3[:, kk] + h / 2 * (np.eye(d) + J_opt3) @ gradfunc(Yopt3[:, kk], X) + np.sqrt(h) * np.random.randn(d)
        Yopt3all[:, :, mm] = Yopt3

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time

    burnin = int(50 / h)
    # burnin = 0


    # Constants
    mupost = [-0.144156092153317, -0.823277471741954]
    mu2post = [0.046915185770836, 1.572054087744374]
    sigmapost = [0.877452061639374, 5.163961990285319]
    sigma2post = [0.784862015909883, 27.182219200533623]
    firstmoment = np.sum([mupost, sigmapost])
    secondmoment = np.sum([mu2post, sigma2post])

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)

    Y_list = [Yall, Yrm2all, Yir1all, Yir2all, Yir3all, Ynopt1all, Ynopt2all, Ynopt3all, Yopt1all, Yopt2all, Yopt3all]
    methods = ['Standard', 'RM_const', 'irr_1', 'irr_2', 'irr_3', 'nopt_1', 'nopt_2', 'nopt_3', 'opt_1', 'opt_2', 'opt_3']
    colors = ['black', '#1a80bb', '#a00000', '#a00000', '#a00000', '#ea801c', '#ea801c', '#ea801c', '#f2c45f', '#f2c45f', '#f2c45f']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # Initialize estimators
    # Calculate number of steps with step_size
    base = 2
    length = int(np.floor(np.log2(K - burnin)))
    x_list = [base ** i for i in range(1, length+1)] + [K-burnin]
    
    estimatorfirst = np.zeros((len(Y_list), length+1, M))
    estimatorsec = np.zeros((len(Y_list), length+1, M)) 
    start_time = time.time()

    for idx, Y in enumerate(Y_list):
        Y_slice = Y[:, burnin:, :]
        for i in range(len(x_list)):
            estimatorfirst[idx, i, :] = np.mean(Y_slice.sum(axis=0)[:x_list[i], :], axis=0)
            estimatorsec[idx, i, :] = np.mean((Y_slice ** 2).sum(axis=0)[:x_list[i], :], axis=0)
        print(np.mean(estimatorfirst[idx,-1,:]))
        print(np.mean(estimatorsec[idx,-1,:]))

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
    np.save(f'{path}/x_list_T{T}_M{M}_h{h}.npy', x_list)
    np.save(f'{path}/mse_first_T{T}_M{M}_h{h}.npy', mse_first)
    np.save(f'{path}/bias_first_sq_T{T}_M{M}_h{h}.npy', bias_first_sq)
    np.save(f'{path}/mse_sec_T{T}_M{M}_h{h}.npy', mse_sec)
    np.save(f'{path}/bias_sec_sq_T{T}_M{M}_h{h}.npy', bias_sec_sq)


    # End timing and store execution time
    end_time = time.time()
    computing_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")
    print(f"Computing Estimators: {estimating_time:.2f} seconds")
    print(f"Computing MSE and Bias: {computing_time:.2f} seconds")

    # Plotting

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # First moment MSE plot
    for idx in range(len(Y_list)):
        ax1.loglog(mse_first[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax1.set_xlabel('steps')
    ax1.set_ylabel('MSE')
    ax1.set_title('First Moment MSE')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.2)
    ax1.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment MSE plot  
    for idx in range(len(Y_list)):
        ax2.loglog(mse_sec[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax2.set_xlabel('steps')
    ax2.set_ylabel('MSE')
    ax2.set_title('Second Moment MSE')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.2)
    ax2.grid(True, which='minor', ls=':', alpha=0.2)

    # First moment bias plot
    for idx in range(len(Y_list)):
        ax3.loglog(bias_first_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax3.set_xlabel('steps')
    ax3.set_ylabel('Squared Bias')
    ax3.set_title('First Moment Squared Bias')
    ax3.legend()
    ax3.grid(True, which='both', ls='-', alpha=0.2)
    ax3.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment bias plot
    for idx in range(len(Y_list)):
        ax4.loglog(bias_sec_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax4.set_xlabel('steps')
    ax4.set_ylabel('Squared Bias')
    ax4.set_title('Second Moment Squared Bias')
    ax4.legend()
    ax4.grid(True, which='both', ls='-', alpha=0.2)
    ax4.grid(True, which='minor', ls=':', alpha=0.2)

    plt.tight_layout()
    plt.title(f'Gaussian parameter estimation with sigma={sigma_true}')
    plt.savefig(f'{path}/mse_bias_3_J_T{T}_M{M}_h{h}.png')
    # plt.show()

    grid_level = 101
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    x = np.linspace(-7, 6, grid_level)  # Wider x range
    y = np.linspace(2, 22, grid_level)  # Wider y range
    X_grid, Y_grid = np.meshgrid(x, y)
    # Compute density
    Z = np.zeros_like(X_grid)
    for i in range(len(x)):
        for j in range(len(y)):
            mu = x[i]
            sigma = y[j]
            Z[j,i] = 1/sigma**N * np.exp(np.sum(-(X[:,0] - mu)**2 / (2 * sigma**2)))

    # Plot contours of density
    contour1 = ax1.contour(x, y, Z, levels=10)
    ax1.set_xlabel('mu')
    ax1.set_ylabel('sigma')
    plt.colorbar(contour1, ax=ax1)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax1.plot(trajectory[0,:4000,0], trajectory[1,:4000,0], label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax1.legend()


    # Plot contours of density
    contour2 = ax2.contour(x, y, Z, levels=10)
    ax2.set_xlabel('mu')
    ax2.set_ylabel('sigma')
    plt.colorbar(contour2, ax=ax2)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax2.plot(np.mean(trajectory[0,:4000,:], axis=1), np.mean(trajectory[1,:4000,:], axis=1), label=methods[idx], linewidth=1, color=colors[idx])
    ax2.legend()


    x = np.linspace(-10, 8, grid_level)  # Wider x range
    y = np.linspace(5, 45, grid_level)  # Wider y range
    X_grid, Y_grid = np.meshgrid(x, y)
    # Compute density
    Z = np.zeros_like(X_grid)
    for i in range(len(x)):
        for j in range(len(y)):
            mu = x[i]
            sigma = y[j]
            Z[j,i] = 1/sigma**N * np.exp(np.sum(-(X[:,1] - mu)**2 / (2 * sigma**2)))

    # Plot contours of density
    contour3 = ax3.contour(x, y, Z, levels=10)
    ax3.set_xlabel('mu')
    ax3.set_ylabel('sigma')
    plt.colorbar(contour3, ax=ax3)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax3.plot(trajectory[2,:4000,0], trajectory[3,:4000,0], label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax3.legend()


    # Plot contours of density
    contour4 = ax4.contour(x, y, Z, levels=10)
    ax4.set_xlabel('mu')
    ax4.set_ylabel('sigma')
    plt.colorbar(contour4, ax=ax4)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax4.plot(np.mean(trajectory[2,:4000,:], axis=1), np.mean(trajectory[3,:4000,:], axis=1), label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax4.legend()

    plt.suptitle(f'trajectory plot, different J, T={T}, M={M}, h={h}')
    plt.savefig(f'trajectory_3_J_T{T}_M{M}_h{h}.png')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.001)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, path=args.path)


