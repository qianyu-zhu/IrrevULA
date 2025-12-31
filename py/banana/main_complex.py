import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from getoptJ import getJ, getnoptJ, getoptJ
from integration import plot_density_with_trajectory, log_banana_density, grad_log_banana_density

# # Parallel processing using multiprocessing
# from multiprocessing import Pool
# import multiprocessing as mp



# --- Langevin MCMC (MALA) ---
def main(M, h, T, path, b = 0.3, c = 0.0, sigma1 = 1.0, sigma2 = 0.3):
    # Check if path exists, if not create it
    if path and not os.path.exists(path):
        os.makedirs(path)

    # Set random seed
    np.random.seed(42)

    fisher = plot_density_with_trajectory(None, b=b, c=c, sigma1=sigma1, sigma2=sigma2, xlim=(-4, 4), ylim=(-4, 4), grid_size=200)

    d = 2
    # Discretization
    K = int(T / h) + 1 # number of steps
    initcond = np.array([5, 5]) # initial condition, true value is [-0.7, 4.4, -1.6, 10.3]
    x = initcond

    firstmoment = np.sum(np.array([0, -b * sigma1**2]))
    secondmoment = np.sum(np.array([sigma1**2, sigma2**2 + 3 * b**2 * sigma1**4]))
    # covariance = 0

    inv_fisher = np.linalg.inv(fisher)
    print('fisher', np.round(fisher, decimals=3))
    J1, J2, J3 = [getJ(fisher) for _ in range(3)]
    J_nopt1, J_nopt2, J_nopt3 = [getnoptJ(fisher) for _ in range(3)]
    J_opt1, J_opt2, J_opt3 = [getoptJ(fisher) for _ in range(3)]


    print('spectrum of random J')
    for J in [J1, J2, J3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
        # print('(J@fisher@J)', np.round(np.trace(J@fisher@J), decimals=3))
        # print('(J@fisher@J@fisher)', np.round(np.trace(J@fisher@J@fisher), decimals=3))
    print('spectrum of nopt J')
    for J in [J_nopt1, J_nopt2, J_nopt3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
        # print('(J@fisher@J)', np.round(np.trace(J@fisher@J), decimals=3))
        # print('(J@fisher@J@fisher)', np.round(np.trace(J@fisher@J@fisher), decimals=3))
    print('spectrum of opt J')
    for J in [J_opt1, J_opt2, J_opt3]:
        print('(I+J)@fisher', np.round(np.linalg.eigvals((np.eye(d) + J)@fisher), decimals=3))
        # print('(J@fisher@J)', np.round(np.trace(J@fisher@J), decimals=3))
        # print('(J@fisher@J@fisher)', np.round(np.trace(J@fisher@J@fisher), decimals=3))
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
            gradeval = gradfunc(Yrm2[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Yrm2[:, kk + 1] = Yrm2[:, kk] + h / 2 * (inv_fisher @ gradeval) + sqrtm(h * inv_fisher) @ np.random.randn(d)
        Yrm2all[:, :, mm] = Yrm2

        # Irreversible Langevin
        Yir1 = np.zeros((d, K))
        Yir1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir1[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Yir1[:, kk + 1] = Yir1[:, kk] + h / 2 * (np.eye(d) + J1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir1all[:, :, mm] = Yir1

        Yir2 = np.zeros((d, K))
        Yir2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir2[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Yir2[:, kk + 1] = Yir2[:, kk] + h / 2 * (np.eye(d) + J2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir2all[:, :, mm] = Yir2

        # Yir3 = np.zeros((d, K))
        # Yir3[:, 0] = initcond
        # for kk in range(K - 1):
        #     gradeval = gradfunc(Yir3[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
        #     Yir3[:, kk + 1] = Yir3[:, kk] + h / 2 * (np.eye(d) + J3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        # Yir3all[:, :, mm] = Yir3

        Ynopt1 = np.zeros((d, K))
        Ynopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt1[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Ynopt1[:, kk + 1] = Ynopt1[:, kk] + h / 2 * (np.eye(d) + J_nopt1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt1all[:, :, mm] = Ynopt1

        Ynopt2 = np.zeros((d, K))
        Ynopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt2[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Ynopt2[:, kk + 1] = Ynopt2[:, kk] + h / 2 * (np.eye(d) + J_nopt2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt2all[:, :, mm] = Ynopt2

        # Ynopt3 = np.zeros((d, K))
        # Ynopt3[:, 0] = initcond
        # for kk in range(K - 1):
        #     gradeval = gradfunc(Ynopt3[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
        #     Ynopt3[:, kk + 1] = Ynopt3[:, kk] + h / 2 * (np.eye(d) + J_nopt3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        # Ynopt3all[:, :, mm] = Ynopt3

        # Optimal Langevin
        Yopt1 = np.zeros((d, K))
        Yopt1[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yopt1[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Yopt1[:, kk + 1] = Yopt1[:, kk] + h / 2 * (np.eye(d) + J_opt1) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yopt1all[:, :, mm] = Yopt1

        Yopt2 = np.zeros((d, K))
        Yopt2[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yopt2[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
            Yopt2[:, kk + 1] = Yopt2[:, kk] + h / 2 * (np.eye(d) + J_opt2) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yopt2all[:, :, mm] = Yopt2

        # Yopt3 = np.zeros((d, K))
        # Yopt3[:, 0] = initcond
        # for kk in range(K - 1):
        #     gradeval = gradfunc(Yopt3[:, kk], b=b, c=c, sigma1=sigma1, sigma2=sigma2)
        #     Yopt3[:, kk + 1] = Yopt3[:, kk] + h / 2 * (np.eye(d) + J_opt3) @ gradeval + np.sqrt(h) * np.random.randn(d)
        # Yopt3all[:, :, mm] = Yopt3
    # End timing and store execution time

    end_time = time.time()
    running_time = end_time - start_time

    burnin = int(10 / h)
    # burnin = 0

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)

    Y_list = [Yall, Yrm2all, Yir1all, Yir2all, Ynopt1all, Ynopt2all, Yopt1all, Yopt2all]
    methods = ['Standard', 'RM_const', 'irr_1', 'irr_2', 'nopt_1', 'nopt_2', 'opt_1', 'opt_2']
    colors = ['black', '#1a80bb', '#a00000', '#a00000', '#ea801c', '#ea801c', '#f2c45f', '#f2c45f']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # Initialize estimators
    # Calculate number of steps with step_size
    base = 1.25
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

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for title
    plt.suptitle(f'Banana with b={b}, c={c}, sigma1={sigma1}, sigma2={sigma2}', y=0.98)
    plt.savefig(f'{path}/mse_bias_Banana_b{b}_T{T}_M{M}_h{h}.png')

    # Plot trajectories
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    grid_level = 101
    
    # Set up grid for contour plot
    x_range = (-4, 4)
    y_range = (-4, 4)
    x = np.linspace(*x_range, grid_level)
    y = np.linspace(*y_range, grid_level)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Calculate density values
    Z = np.zeros_like(X_grid)
    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):
            Z[i,j] = np.exp(log_banana_density([X_grid[i,j], Y_grid[i,j]], b=b, c=c, sigma1=sigma1, sigma2=sigma2))
            
    # Plot single trajectory with contour
    contour1 = axes2[0].contour(x, y, Z, levels=15, alpha=0.4)
    plt.colorbar(contour1, ax=axes2[0])
    for idx, trajectory in enumerate(Y_list):
        axes2[0].plot(trajectory[0,:500,0], trajectory[1,:500,0],
                     label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    axes2[0].set_xlabel('x1')
    axes2[0].set_ylabel('x2') 
    axes2[0].set_title('Single Trajectory')
    axes2[0].legend()

    # Plot averaged trajectory with contour
    contour2 = axes2[1].contour(x, y, Z, levels=15, alpha=0.4)
    plt.colorbar(contour2, ax=axes2[1])
    for idx, trajectory in enumerate(Y_list):
        axes2[1].plot(np.mean(trajectory[0,:500,:], axis=1), np.mean(trajectory[1,:500,:], axis=1),
                     label=methods[idx], linewidth=1, color=colors[idx])
    axes2[1].set_xlabel('x1')
    axes2[1].set_ylabel('x2')
    axes2[1].set_title('Averaged Trajectory')
    axes2[1].legend()



    plt.suptitle(f'trajectory plot on Banana b={b}, c={c}, sigma1={sigma1}, sigma2={sigma2}, T={T}, M={M}, h={h}')
    plt.savefig(f'{path}/trajectory_Banana_b{b}_T{T}_M{M}_h{h}.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--h', type=float, default=0.01)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--b', type=float, default=0.3)
    parser.add_argument('--c', type=float, default=0.0)
    parser.add_argument('--sigma1', type=float, default=1.0)
    parser.add_argument('--sigma2', type=float, default=0.3)
    parser.add_argument('--path', type=str, default='banana_complex')
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T, path=args.path, b=args.b, c=args.c, sigma1=args.sigma1, sigma2=args.sigma2)