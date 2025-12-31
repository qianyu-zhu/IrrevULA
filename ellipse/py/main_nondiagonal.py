import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import special_ortho_group

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ, getJ
from adaptfisher import initialize_A1, iterate_An, initialize_FIM, iterate_FIM


"""
This script is used to run the simulation for the 4-D Gaussian with non-diagonal covariance.
"""

def main(M,h,T,sigma_true,path):
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Setup
    d = 4
    K = int(T / h) + 1 # number of steps
    mu_true = [0 for _ in range(d)]
    # non-diagonal covariance
    R = special_ortho_group.rvs(dim=d)
    cov_true = R@ (np.diag(sigma_true)**2) @R.T
    fisher = np.linalg.inv(cov_true)
    initcond = np.array([5, 5, 5, 5]) # initial condition, true value is [-0.7, 4.4, -1.6, 10.3]
    scale = np.trace(fisher) / np.trace(cov_true)

    print('parameters: T =', T,
          'dt =', h,
          'd =', d,
          'number of chains =', M,
          'number of steps =', K)
    # Generate 5 random skew-symmetric matrices
    print('fisher', np.round(fisher, decimals=3))


    # Initialize arrays
    Yall = np.zeros((d, K, M)) # dimension * stepsize * chain
    Yrmall1 = np.zeros((d, K, M))
    Yirall1 = np.zeros((d, K, M))
    Ynoptall1 = np.zeros((d, K, M))
    Yoptall1 = np.zeros((d, K, M))

        
    def simulate_chain(mm):
        np.random.seed(mm)  # for reproducibility in parallel
        
        Y = np.zeros((d, K))
        Y[:, 0] = initcond
        for kk in range(K - 1):
            Y[:, kk + 1] = Y[:, kk] + h / 2 * gradfunc(Y[:, kk], fisher) + np.sqrt(h) * np.random.randn(d)

        # RM Langevin constant
        Yrm = np.zeros((d, K))
        fisher_error0 = np.zeros((K,))
        Yrm[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yrm[:, kk], fisher)
            if kk == 0:
                FIM_inv = initialize_A1(gradeval)
            else:
                FIM_inv = iterate_An(FIM_inv, gradeval, kk)
            Yrm[:, kk + 1] = Yrm[:, kk] + h / 2 * (FIM_inv) @ gradeval + np.linalg.cholesky(h * FIM_inv) @ np.random.randn(d)
            fisher_error0[kk] = np.linalg.norm(cov_true - FIM_inv)

        # Irreversible Langevin
        Yir = np.zeros((d, K))
        fisher_error1 = np.zeros((K,))
        Yir[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir[:, kk], fisher)
            if kk == 0:
                FIM = initialize_FIM(gradeval)
            else:
                FIM = iterate_FIM(FIM, gradeval, kk)
            J1 = getJ(FIM)
            Yir[:, kk + 1] = Yir[:, kk] + h / 2 * (np.eye(d) + J1) @ gradeval + np.sqrt(h) * np.random.randn(d)
            fisher_error1[kk] = np.linalg.norm(FIM - fisher)

            
        # Irreversible Langevin, non-optimal preconditioner
        Ynopt = np.zeros((d, K))
        fisher_error2 = np.zeros((K,))
        Ynopt[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt[:, kk], fisher)
            if kk == 0:
                FIM = initialize_FIM(gradeval)
            else:
                FIM = iterate_FIM(FIM, gradeval, kk)
            J_nopt = getnoptJ(FIM)
            Ynopt[:, kk + 1] = Ynopt[:, kk] + h / 2 * (np.eye(d) + J_nopt) @ gradeval + np.sqrt(h) * np.random.randn(d)
            fisher_error2[kk] = np.linalg.norm(FIM - fisher)

        # Optimal Langevin
        Yopt = np.zeros((d, K))
        fisher_error3 = np.zeros((K,))
        Yopt[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yopt[:, kk], fisher)
            if kk == 0:
                FIM = initialize_FIM(gradeval)
            else:
                FIM = iterate_FIM(FIM, gradeval, kk)
            J_opt = getoptJ(FIM)
            Yopt[:, kk + 1] = Yopt[:, kk] + h / 2 * (np.eye(d) + J_opt) @ gradeval + np.sqrt(h) * np.random.randn(d)
            fisher_error3[kk] = np.linalg.norm(FIM - fisher)

        return Y, Yrm, Yir, Ynopt, Yopt, fisher_error0, fisher_error1, fisher_error2, fisher_error3  # list of shape [11 arrays of shape (d, K)]
        

    print("Running simulations in parallel...")
    start_time = time.time()
    num_cores = 50 #multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(
        delayed(simulate_chain)(mm) for mm in tqdm(range(M))
    )

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time

    # Extract results from valid chains
    burnin = int(50 / h)
    # burnin = 0

    Yall, Yrmall, Yirall, Ynoptall, Yoptall, fisher_error0, fisher_error1, fisher_error2, fisher_error3 = [np.array(x) for x in zip(*results)]
    Yall = Yall.transpose(1, 2, 0)
    Yrmall = Yrmall.transpose(1, 2, 0)
    Yirall = Yirall.transpose(1, 2, 0)
    Ynoptall = Ynoptall.transpose(1, 2, 0)
    Yoptall = Yoptall.transpose(1, 2, 0)
    fisher_error0 = fisher_error0.mean(axis=0)
    fisher_error1 = fisher_error1.mean(axis=0)
    fisher_error2 = fisher_error2.mean(axis=0)
    fisher_error3 = fisher_error3.mean(axis=0)
    print(Yall.shape)

    plt.plot(fisher_error0 * scale, label='RM')
    plt.plot(fisher_error1, label='irr')
    plt.plot(fisher_error2, label='nopt')
    plt.plot(fisher_error3, label='opt')
    plt.legend()
    plt.savefig(f'{path}/fisher_convergence_T{T}_M{M}_h{h}.png')  


    # Constants
    firstmoment = 0
    secondmoment = np.sum(np.diag(cov_true))

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)

    Y_list = [Yall, Yrmall, Yirall, Ynoptall, Yoptall]
    methods = ['Standard', 'RM', 'irr', 'nopt', 'opt']
    colors = ['black', '#1a80bb', '#a00000', '#ea801c', '#f2c45f']
    alphas = [1, 1, 1, 1, 1]
    # Initialize estimators
    # Calculate number of steps with step_size
    step_size = 10
    num_steps = (K - burnin) // step_size
    
    estimatorfirst = np.zeros((len(Y_list), num_steps, M))
    estimatorsec = np.zeros((len(Y_list), num_steps, M))

    # Start timing
    start_time = time.time()
    # Compute estimators
    for idx, Y in enumerate(Y_list):
        for ii in range(M): # for each chain
            Y_slice = Y[:, burnin:, ii] # burnin is the number of steps to discard
            denom = np.arange(1, K - burnin + 1)[step_size-1::step_size]
            NN = Y_slice.shape[1]
            estimatorfirst[idx, :, ii] = np.cumsum(Y_slice.sum(axis=0)[:NN - (NN % step_size)].reshape(-1, step_size).sum(axis=1)) / denom
            estimatorsec[idx, :, ii] = np.cumsum((Y_slice ** 2).sum(axis=0)[:NN - (NN % step_size)].reshape(-1, step_size).sum(axis=1)) / denom

        # print(methods[idx], '--------------------------------')
        print(np.mean(estimatorfirst[idx,-1,:]))
        print(np.mean(estimatorsec[idx,-1,:]))

    # End timing and store execution time
    end_time = time.time()
    estimating_time = end_time - start_time

    # Start timing
    start_time = time.time()
    # Compute Mean Squared Errors and Biases
    mse_first = np.zeros((len(Y_list), num_steps))
    bias_first_sq = np.zeros((len(Y_list), num_steps))
    for idx in range(len(Y_list)):
        mse = np.mean((estimatorfirst[idx] - firstmoment) ** 2, axis=1)
        biassq = np.mean(estimatorfirst[idx] - firstmoment, axis=1) ** 2
        mse_first[idx] = mse
        bias_first_sq[idx] = biassq

    # Second moment MSE calculations commented out
    mse_sec = np.zeros((len(Y_list), num_steps))
    bias_sec_sq = np.zeros((len(Y_list), num_steps))
    for idx in range(len(Y_list)):
        mse = np.mean((estimatorsec[idx] - secondmoment) ** 2, axis=1)
        biassq = np.mean(estimatorsec[idx] - secondmoment, axis=1) ** 2
        mse_sec[idx] = mse
        bias_sec_sq[idx] = biassq

    # End timing and store execution time
    end_time = time.time()
    computing_time = end_time - start_time

    print(f"Running time: {running_time:.2f} seconds")
    print(f"Computing Estimators: {estimating_time:.2f} seconds")
    print(f"Computing MSE and Bias: {computing_time:.2f} seconds")


    print(f"Running time: {running_time:.2f} seconds")
    print(f"Computing Estimators: {estimating_time:.2f} seconds")
    print(f"Computing MSE and Bias: {computing_time:.2f} seconds")


    # Plotting

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # First moment MSE plot
    for idx in range(len(Y_list)):
        ax1.loglog(mse_first[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax1.set_yscale('log')
    ax1.set_xlabel('10^3 steps')
    ax1.set_ylabel('MSE')
    ax1.set_title('First Moment MSE')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.2)
    ax1.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment MSE plot  
    for idx in range(len(Y_list)):
        ax2.loglog(mse_sec[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax2.set_yscale('log')
    ax2.set_xlabel('10^3 steps')
    ax2.set_ylabel('MSE')
    ax2.set_title('Second Moment MSE')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.2)
    ax2.grid(True, which='minor', ls=':', alpha=0.2)

    # First moment bias plot
    for idx in range(len(Y_list)):
        ax3.loglog(bias_first_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax3.set_yscale('log')
    ax3.set_xlabel('10^3 steps')
    ax3.set_ylabel('Squared Bias')
    ax3.set_title('First Moment Squared Bias')
    ax3.legend()
    ax3.grid(True, which='both', ls='-', alpha=0.2)
    ax3.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment bias plot
    for idx in range(len(Y_list)):
        ax4.loglog(bias_sec_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax4.set_yscale('log')
    ax4.set_xlabel('10^3 steps')
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
    x = np.linspace(-10, 10, grid_level)  # Wider x range
    y = np.linspace(-10, 10, grid_level)  # Wider y range
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


    x = np.linspace(-30, 30, grid_level)  # Wider x range
    y = np.linspace(-30, 30, grid_level)  # Wider y range
    X_grid, Y_grid = np.meshgrid(x, y)
    # Compute density
    Z = np.zeros_like(X_grid)
    for i in range(len(x)):
        for j in range(len(y)):
            X = np.array([x[i], y[j]])
            Z[j,i] = 1/ np.prod(sigma_true[-2:]) * np.exp(-X@fisher[-2:, -2:]@X / 2)

    # Plot contours of density
    contour3 = ax3.contour(x, y, Z, levels=10)
    ax3.set_xlabel('mu')
    ax3.set_ylabel('sigma')
    plt.colorbar(contour3, ax=ax3)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax3.plot(trajectory[-2,:4000,0], trajectory[-1,:4000,0], label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax3.legend()


    # Plot contours of density
    contour4 = ax4.contour(x, y, Z, levels=10)
    ax4.set_xlabel('mu')
    ax4.set_ylabel('sigma')
    plt.colorbar(contour4, ax=ax4)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax4.plot(np.mean(trajectory[-2,:4000,:], axis=1), np.mean(trajectory[-1,:4000,:], axis=1), label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax4.legend()


    plt.suptitle(f'4-D Gaussian trajectory plot with different J\nT={T}, M={M}, h={h}, covariance={np.diag(cov_true)}')
    plt.savefig(f'{path}/trajectory_4D_Gaussian_T{T}_M{M}_h{h}.png')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.04)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--sigma_true', type=float, nargs='+', default=[2**i for i in range(4)])
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, sigma_true=args.sigma_true, path=args.path)


