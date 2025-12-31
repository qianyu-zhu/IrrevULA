import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ, getJ

"""
This script is unparalleled.
"""

def main(M,h,T,sigma_true,path):
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Setup
    d = 4
    M = 100
    h = 0.04
    T = 50000
    K = int(T / h) + 1 # number of steps
    mu_true = [0 for _ in range(d)]
    cov_true = np.diag(sigma_true)**2
    fisher = np.linalg.inv(cov_true)
    initcond = np.array([0, 0, 0, 0]) # initial condition, true value is [-0.7, 4.4, -1.6, 10.3]

    print('parameters: T =', T,
          'dt =', h,
          'd =', d,
          'number of chains =', M,
          'number of steps =', K)
    # Generate 5 random skew-symmetric matrices
    print('fisher', np.round(fisher, decimals=3))
    J1, J2, J3 = [getJ(fisher) for _ in range(3)]
    J_nopt1, J_nopt2, J_nopt3 = [getnoptJ(fisher) for _ in range(3)]
    J_opt1, J_opt2, J_opt3 = [getoptJ(fisher) for _ in range(3)]


    print('spectrum of random J')
    for J in [J1, J2, J3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
    print('spectrum of nopt J')
    for J in [J_nopt1, J_nopt2, J_nopt3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
    print('spectrum of opt J')
    for J in [J_opt1, J_opt2, J_opt3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))

    # Initialize arrays
    Yall = np.zeros((d, K, M)) # dimension * stepsize * chain
    Yrm2all = np.zeros((d, K, M))

    Yir1all, Yir2all, Yir3all = [np.zeros((d, K, M)) for _ in range(3)]
    Ynopt1all, Ynopt2all, Ynopt3all = [np.zeros((d, K, M)) for _ in range(3)]
    Yopt1all, Yopt2all, Yopt3all = [np.zeros((d, K, M)) for _ in range(3)]

    # Start timing


    for mm in tqdm(range(M)):
        # Standard Langevin
        Y = np.zeros((d, K))
        Y[:, 0] = initcond
        for kk in range(K - 1):
            Y[:, kk + 1] = Y[:, kk] + h / 2 * gradfunc(Y[:, kk], fisher) + np.sqrt(h) * np.random.randn(d)
        Yall[:, :, mm] = Y

        # RM Langevin constant
        Yrm2 = np.zeros((d, K))
        Yrm2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yrm2[:, kk], fisher)
            Yrm2[:, kk + 1] = Yrm2[:, kk] + h / 2 * (cov_true @ gradeval) + sqrtm(h * cov_true) @ np.random.randn(d)
        Yrm2all[:, :, mm] = Yrm2

        # Irreversible Langevin
        Yir1 = np.zeros((d, K))
        Yir1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir1[:, kk], fisher)
            Yir1[:, kk + 1] = Yir1[:, kk] + h / 2 * (np.eye(d) + J1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir1all[:, :, mm] = Yir1

        Yir2 = np.zeros((d, K))
        Yir2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir2[:, kk], fisher)
            Yir2[:, kk + 1] = Yir2[:, kk] + h / 2 * (np.eye(d) + J2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir2all[:, :, mm] = Yir2

        Yir3 = np.zeros((d, K))
        Yir3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir3[:, kk], fisher)
            Yir3[:, kk + 1] = Yir3[:, kk] + h / 2 * (np.eye(d) + J3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir3all[:, :, mm] = Yir3


        Ynopt1 = np.zeros((d, K))
        Ynopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt1[:, kk], fisher)
            Ynopt1[:, kk + 1] = Ynopt1[:, kk] + h / 2 * (np.eye(d) + J_nopt1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt1all[:, :, mm] = Ynopt1

        Ynopt2 = np.zeros((d, K))
        Ynopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt2[:, kk], fisher)
            Ynopt2[:, kk + 1] = Ynopt2[:, kk] + h / 2 * (np.eye(d) + J_nopt2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt2all[:, :, mm] = Ynopt2

        Ynopt3 = np.zeros((d, K))
        Ynopt3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt3[:, kk], fisher)
            Ynopt3[:, kk + 1] = Ynopt3[:, kk] + h / 2 * (np.eye(d) + J_nopt3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt3all[:, :, mm] = Ynopt3


        # Optimal Langevin
        Yopt1 = np.zeros((d, K))
        Yopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yopt1[:, kk], fisher)
            Yopt1[:, kk + 1] = Yopt1[:, kk] + h / 2 * (np.eye(d) + J_opt1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yopt1all[:, :, mm] = Yopt1

        Yopt2 = np.zeros((d, K))
        Yopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yopt2[:, kk], fisher)
            Yopt2[:, kk + 1] = Yopt2[:, kk] + h / 2 * (np.eye(d) + J_opt2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yopt2all[:, :, mm] = Yopt2

        Yopt3 = np.zeros((d, K))
        Yopt3[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yopt3[:, kk], fisher)
            Yopt3[:, kk + 1] = Yopt3[:, kk] + h / 2 * (np.eye(d) + J_opt3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yopt3all[:, :, mm] = Yopt3


    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time

    burnin = int(50 / h)
    # burnin = 0


    # Constants
    firstmoment = 0
    secondmoment = np.sum(np.diag(cov_true))

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)

    Y_list = [Yall, Yrm2all, Yir1all, Yir2all, Yir3all, Ynopt1all, Ynopt2all, Ynopt3all, Yopt1all, Yopt2all, Yopt3all]
    methods = ['Standard', 'RM_const', 'irr_1', 'irr_2', 'irr_3', 'nopt_1', 'nopt_2', 'nopt_3', 'opt_1', 'opt_2', 'opt_3']
    colors = ['black', '#1a80bb', '#a00000', '#a00000', '#a00000', '#ea801c', '#ea801c', '#ea801c', '#f2c45f', '#f2c45f', '#f2c45f']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # Initialize estimators
    # Calculate number of steps with step_size
    base = 1.5
    length = int(np.floor(np.log(K - burnin) / np.log(base)))
    x_list = [int(base ** i) for i in range(1, length+1)] + [K-burnin]
    
    estimatorfirst = np.zeros((len(Y_list), length+1, M))
    estimatorsec = np.zeros((len(Y_list), length+1, M)) 
    start_time = time.time()

    for idx, Y in enumerate(Y_list):
        Y_slice = Y[:, burnin:, :]
        for i in range(len(x_list)):
            estimatorfirst[idx, i, :] = np.mean(Y_slice.sum(axis=0)[:x_list[i], :], axis=0)
            estimatorsec[idx, i, :] = np.mean((Y_slice ** 2).sum(axis=0)[:x_list[i], :], axis=0)
    # estimatorfirst = np.zeros((len(Y_list), num_steps, M))
    # estimatorsec = np.zeros((len(Y_list), num_steps, M))

    # # Start timing
    # start_time = time.time()

    # # Compute estimators
    # for idx, Y in enumerate(Y_list):
    #     for ii in range(M): # for each chain
    #         Y_slice = Y[:, burnin:, ii] # burnin is the number of steps to discard
    #         denom = np.arange(1, K - burnin + 1)[step_size-1::step_size]
    #         NN = Y_slice.shape[1]
    #         estimatorfirst[idx, :, ii] = np.cumsum(Y_slice.sum(axis=0)[:NN - (NN % step_size)].reshape(-1, step_size).sum(axis=1)) / denom
    #         estimatorsec[idx, :, ii] = np.cumsum((Y_slice ** 2).sum(axis=0)[:NN - (NN % step_size)].reshape(-1, step_size).sum(axis=1)) / denom

    #     # print(methods[idx], '--------------------------------')
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
        ax1.loglog(x_list, mse_first[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax1.set_xlabel('steps')
    ax1.set_ylabel('MSE')
    ax1.set_title('First Moment MSE')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.2)
    ax1.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment MSE plot  
    for idx in range(len(Y_list)):
        ax2.loglog(x_list, mse_sec[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax2.set_xlabel('steps')
    ax2.set_ylabel('MSE')
    ax2.set_title('Second Moment MSE')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.2)
    ax2.grid(True, which='minor', ls=':', alpha=0.2)

    # First moment bias plot
    for idx in range(len(Y_list)):
        ax3.loglog(x_list, bias_first_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax3.set_xlabel('steps')
    ax3.set_ylabel('Squared Bias')
    ax3.set_title('First Moment Squared Bias')
    ax3.legend()
    ax3.grid(True, which='both', ls='-', alpha=0.2)
    ax3.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment bias plot
    for idx in range(len(Y_list)):
        ax4.loglog(x_list, bias_sec_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax4.set_xlabel('steps')
    ax4.set_ylabel('Squared Bias')
    ax4.set_title('Second Moment Squared Bias')
    ax4.legend()
    ax4.grid(True, which='both', ls='-', alpha=0.2)
    ax4.grid(True, which='minor', ls=':', alpha=0.2)

    plt.tight_layout()
    plt.title(f'4-D Gaussian with covariance {np.diag(cov_true)}')
    plt.savefig(f'{path}/mse_bias_4D_Gaussian_T{T}_M{M}_h{h}.png')
    # plt.show()

    grid_level = 101
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    x = np.linspace(-20, 20, grid_level)  # Wider x range
    y = np.linspace(-20, 20, grid_level)  # Wider y range
    X_grid, Y_grid = np.meshgrid(x, y)
    # Compute density
    Z = np.zeros_like(X_grid)
    for i in range(len(x)):
        for j in range(len(y)):
            X = np.array([x[i], y[j]])
            Z[j,i] = 1/ np.prod(sigma_true[:2]) * np.exp(-X@fisher[:2,:2]@X / 2)

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


    x = np.linspace(-100, 100, grid_level)  # Wider x range
    y = np.linspace(-100, 100, grid_level)  # Wider y range
    X_grid, Y_grid = np.meshgrid(x, y)
    # Compute density
    Z = np.zeros_like(X_grid)
    for i in range(len(x)):
        for j in range(len(y)):
            X = np.array([x[i], y[j]])
            Z[j,i] = 1/ np.prod(sigma_true[2:]) * np.exp(-X@fisher[2:,2:]@X / 2)

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

    plt.suptitle(f'4-D Gaussian trajectory plot with different J\nT={T}, M={M}, h={h}, covariance={np.diag(cov_true)}')
    plt.savefig(f'{path}/trajectory_4D_Gaussian_T{T}_M{M}_h{h}.png')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.001)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--sigma_true', type=float, nargs='+', default=[2**i for i in range(4)])
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, sigma_true=args.sigma_true, path=args.path)


