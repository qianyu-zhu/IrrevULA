import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from gradfunc import gradfunc
from getoptJ import getoptJ, getnoptJ

def main(M,h,T):
    # Set random seed
    np.random.seed(42)

    # Data
    N = 30
    mu_true = [0, 0]
    sigma_true = [5, 10]
    X = mu_true + sigma_true * np.random.randn(N, 2) # random initialization
    d = X.shape[1]*2

    # Discretization
    K = int(T / h) + 1 # number of steps
    initcond = np.array([5, 20, 5, 40]) # initial condition, true value is ~[-0.7, 4.4, -1.6, 10.3]

    # Functions
    B = lambda state: (1 / N) * np.diag([state[1] ** 2, state[1] ** 2 / 2, state[3] ** 2, state[3] ** 2 / 2]) # inverse Fisher
    divB = lambda state: (1 / N) * np.array([0, state[1], 0, state[3]])

    # delta = 1
    # J = delta * np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    # C = lambda state: 0.75 / N * np.array([[0, state[1] ** 2, 0, 0], [-state[1] ** 2, 0, 0, 0], [0, 0, 0, state[3] ** 2], [0, 0, -state[3] ** 2, 0]])
    # divC = lambda state: 3 / 2 * delta  / N * np.array([state[1], 0, state[3], 0])

    # inv_fisher = B([mu_true[0], sigma_true[0], mu_true[1], sigma_true[1]])
    # fisher = np.linalg.inv(inv_fisher)
    fisher = np.load('statistics/fisher_information_500000.npy')
    inv_fisher = np.linalg.inv(fisher)

    def generate_J(d):
        J = np.random.randn(d,d)
        J = (J - J.T)/2
        return J
    np.random.seed()

    def generate_noptJ(fisher):
        i = 0
        while i <= 1000:
            candidate_J = getnoptJ(fisher)
            if np.linalg.norm(candidate_J) >= 5:
                return candidate_J
            i += 1
        return candidate_J

    # Generate 5 random skew-symmetric matrices
    J1 = generate_J(d)
    J2 = generate_J(d)
    J3 = generate_J(d)
    J4 = generate_J(d)
    J5 = generate_J(d)
    J_nopt1 = getnoptJ(fisher)
    J_nopt2 = getnoptJ(fisher)
    J_nopt3 = getnoptJ(fisher)
    J_nopt4 = getnoptJ(fisher)
    J_nopt5 = getnoptJ(fisher)
    J_opt1 = getoptJ(fisher)
    J_opt2 = getoptJ(fisher)
    J_opt3 = getoptJ(fisher)
    J_opt4 = getoptJ(fisher)
    J_opt5 = getoptJ(fisher)

    # print('inv_fisher', inv_fisher)
    print('fisher', np.round(fisher, decimals=3))

    # print('J1', np.round(J1, decimals=3))
    # print('J2', np.round(J2, decimals=3))
    # print('J3', np.round(J3, decimals=3))
    # print('J4', np.round(J4, decimals=3))
    # print('J5', np.round(J5, decimals=3))
    # print('J_opt1', np.round(J_opt1, decimals=3))
    # print('J_opt2', np.round(J_opt2, decimals=3))
    # print('J_opt3', np.round(J_opt3, decimals=3))
    # print('J_opt4', np.round(J_opt4, decimals=3))
    # print('J_opt5', np.round(J_opt5, decimals=3))

    print('(I+J1)@fisher', np.linalg.eigvals((np.eye(d) + J1)@fisher))
    print('(I+J2)@fisher', np.linalg.eigvals((np.eye(d) + J2)@fisher))
    print('(I+J3)@fisher', np.linalg.eigvals((np.eye(d) + J3)@fisher))
    print('(I+J4)@fisher', np.linalg.eigvals((np.eye(d) + J4)@fisher))
    print('(I+J5)@fisher', np.linalg.eigvals((np.eye(d) + J5)@fisher))
    print('(I+J_nopt1)@fisher', np.linalg.eigvals((np.eye(d) + J_nopt1)@fisher))
    print('(I+J_nopt2)@fisher', np.linalg.eigvals((np.eye(d) + J_nopt2)@fisher))
    print('(I+J_nopt3)@fisher', np.linalg.eigvals((np.eye(d) + J_nopt3)@fisher))
    print('(I+J_nopt4)@fisher', np.linalg.eigvals((np.eye(d) + J_nopt4)@fisher))
    print('(I+J_nopt5)@fisher', np.linalg.eigvals((np.eye(d) + J_nopt5)@fisher))
    print('(I+J_opt1)@fisher', np.linalg.eigvals((np.eye(d) + J_opt1)@fisher))
    print('(I+J_opt2)@fisher', np.linalg.eigvals((np.eye(d) + J_opt2)@fisher))
    print('(I+J_opt3)@fisher', np.linalg.eigvals((np.eye(d) + J_opt3)@fisher))
    print('(I+J_opt4)@fisher', np.linalg.eigvals((np.eye(d) + J_opt4)@fisher))
    print('(I+J_opt5)@fisher', np.linalg.eigvals((np.eye(d) + J_opt5)@fisher))

    # Initialize arrays
    Yall = np.zeros((d, K, M)) # dimension * stepsize * chain
    # Yrmall = np.zeros((d, K, M))
    Yrm2all = np.zeros((d, K, M))

    Yir1all = np.zeros((d, K, M))
    Yir2all = np.zeros((d, K, M))   
    Yir3all = np.zeros((d, K, M))
    Yir4all = np.zeros((d, K, M))
    Yir5all = np.zeros((d, K, M))
    Ynopt1all = np.zeros((d, K, M))
    Ynopt2all = np.zeros((d, K, M))
    Ynopt3all = np.zeros((d, K, M))
    Ynopt4all = np.zeros((d, K, M))
    Ynopt5all = np.zeros((d, K, M))
    # Yirmall = np.zeros((d, K, M))
    # Yirrmall = np.zeros((d, K, M))
    Yopt1all = np.zeros((d, K, M))
    Yopt2all = np.zeros((d, K, M))
    Yopt3all = np.zeros((d, K, M))
    Yopt4all = np.zeros((d, K, M))
    Yopt5all = np.zeros((d, K, M))

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

        # # RM Langevin
        # Yrm = np.zeros((d, K))
        # Yrm[:, 0] = initcond
        # for kk in range(K - 1):
        #     Beval = B(Yrm[:, kk])
        #     divBeval = divB(Yrm[:, kk])
        #     gradeval = gradfunc(Yrm[:, kk], X)
        #     Yrm[:, kk + 1] = Yrm[:, kk] + h / 2 * (Beval @ gradeval + divBeval) + np.sqrt(h * Beval) @ np.random.randn(d)
        # Yrmall[:, :, mm] = Yrm

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

        Yir4 = np.zeros((d, K))
        Yir4[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir4[:, kk], X)
            Yir4[:, kk + 1] = Yir4[:, kk] + h / 2 * (np.eye(d) + J4) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir4all[:, :, mm] = Yir4

        Yir5 = np.zeros((d, K))
        Yir5[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Yir5[:, kk], X)
            Yir5[:, kk + 1] = Yir5[:, kk] + h / 2 * (np.eye(d) + J5) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Yir5all[:, :, mm] = Yir5

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

        Ynopt4 = np.zeros((d, K))
        Ynopt4[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt4[:, kk], X)
            Ynopt4[:, kk + 1] = Ynopt4[:, kk] + h / 2 * (np.eye(d) + J_nopt4) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt4all[:, :, mm] = Ynopt4

        Ynopt5 = np.zeros((d, K))
        Ynopt5[:, 0] = initcond
        for kk in range(K - 1):
            gradeval = gradfunc(Ynopt5[:, kk], X)
            Ynopt5[:, kk + 1] = Ynopt5[:, kk] + h / 2 * (np.eye(d) + J_nopt5) @ gradeval + np.sqrt(h) * np.random.randn(d)
        Ynopt5all[:, :, mm] = Ynopt5

        # # Irreversible on RM Langevin
        # Yirm = np.zeros((d, K))
        # Yirm[:, 0] = initcond
        # for kk in range(K - 1):
        #     Beval = B(Yirm[:, kk])
        #     divBeval = divB(Yirm[:, kk])
        #     gradeval = gradfunc(Yirm[:, kk], X)
        #     Yirm[:, kk + 1] = Yirm[:, kk] + h * ((1 / 2 * Beval + J) @ gradeval + 1 / 2 * divBeval) + np.sqrt(h * Beval) @ np.random.randn(d)
        # Yirmall[:, :, mm] = Yirm

        # # Irreversible RM Langevin
        # Yirrm = np.zeros((d, K))
        # Yirrm[:, 0] = initcond
        # for kk in range(K - 1):
        #     Beval = B(Yirrm[:, kk])
        #     Ceval = C(Yirrm[:, kk])
        #     divBeval = divB(Yirrm[:, kk])
        #     divCeval = divC(Yirrm[:, kk])
        #     gradeval = gradfunc(Yirrm[:, kk], X)
        #     Yirrm[:, kk + 1] = Yirrm[:, kk] + h * ((1 / 2 * Beval + Ceval) @ gradeval + 1 / 2 * divBeval + divCeval) + np.sqrt(h * Beval) @ np.random.randn(d)
        # Yirrmall[:, :, mm] = Yirrm

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

        Yopt4 = np.zeros((d, K))
        Yopt4[:, 0] = initcond
        for kk in range(K - 1):
            Yopt4[:, kk + 1] = Yopt4[:, kk] + h / 2 * (np.eye(d) + J_opt4) @ gradfunc(Yopt4[:, kk], X) + np.sqrt(h) * np.random.randn(d)
        Yopt4all[:, :, mm] = Yopt4

        Yopt5 = np.zeros((d, K))
        Yopt5[:, 0] = initcond
        for kk in range(K - 1):
            Yopt5[:, kk + 1] = Yopt5[:, kk] + h / 2 * (np.eye(d) + J_opt5) @ gradfunc(Yopt5[:, kk], X) + np.sqrt(h) * np.random.randn(d)
        Yopt5all[:, :, mm] = Yopt5

    # End timing and store execution time
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")

    burnin = int(50 / h)
    # burnin = 0


    # Constants
    mupost = [-0.720439004756125, -1.651146562434101]
    mu2post = [1.171262321162274, 6.348858222073647]
    sigmapost = [4.387137064233570, 10.330273538213467]
    sigma2post = [19.620296285028772, 108.785453529268580]
    firstmoment = np.sum([mupost, sigmapost])
    secondmoment = np.sum([mu2post, sigma2post])

    print('firstmoment', firstmoment)
    print('secondmoment', secondmoment)

    Y_list = [Yall, Yrm2all, Yir1all, Yir2all, Yir3all, Yir4all, Yir5all, Ynopt1all, Ynopt2all, Ynopt3all, Ynopt4all, Ynopt5all, Yopt1all, Yopt2all, Yopt3all, Yopt4all, Yopt5all]
    methods = ['Standard', 'RM_const', 'irr_1', 'irr_2', 'irr_3', 'irr_4', 'irr_5', 'nopt_1', 'nopt_2', 'nopt_3', 'nopt_4', 'nopt_5', 'opt_1', 'opt_2', 'opt_3', 'opt_4', 'opt_5']
    colors = ['black', '#1a80bb', '#a00000', '#a00000', '#a00000', '#a00000', '#a00000', '#ea801c', '#ea801c', '#ea801c', '#ea801c', '#ea801c', '#f2c45f', '#f2c45f', '#f2c45f', '#f2c45f', '#f2c45f']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # Initialize estimators
    # Calculate number of steps with step_size
    step_size = 1000
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
    print(f"Computing Estimators: {estimating_time:.2f} seconds")


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
    print(f"Computing MSE and Bias: {computing_time:.2f} seconds")

    # Plotting

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # First moment MSE plot
    for idx in range(len(Y_list)):
        ax1.loglog(mse_first[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax1.set_yscale('log')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE')
    ax1.set_title('First Moment MSE')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.2)
    ax1.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment MSE plot  
    for idx in range(len(Y_list)):
        ax2.loglog(mse_sec[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax2.set_yscale('log')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('MSE')
    ax2.set_title('Second Moment MSE')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.2)
    ax2.grid(True, which='minor', ls=':', alpha=0.2)

    # First moment bias plot
    for idx in range(len(Y_list)):
        ax3.loglog(bias_first_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax3.set_yscale('log')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Squared Bias')
    ax3.set_title('First Moment Squared Bias')
    ax3.legend()
    ax3.grid(True, which='both', ls='-', alpha=0.2)
    ax3.grid(True, which='minor', ls=':', alpha=0.2)

    # Second moment bias plot
    for idx in range(len(Y_list)):
        ax4.loglog(bias_sec_sq[idx], label=methods[idx], color=colors[idx], alpha=alphas[idx])
    ax4.set_yscale('log')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Squared Bias')
    ax4.set_title('Second Moment Squared Bias')
    ax4.legend()
    ax4.grid(True, which='both', ls='-', alpha=0.2)
    ax4.grid(True, which='minor', ls=':', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'mse_bias_3_J_T{T}_M{M}_h{h}.png')
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
    contour1 = ax1.contour(x, y, Z, levels=20)
    ax1.set_xlabel('mu')
    ax1.set_ylabel('sigma')
    plt.colorbar(contour1, ax=ax1)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax1.plot(trajectory[0,:4000,0], trajectory[1,:4000,0], label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax1.legend()


    # Plot contours of density
    contour2 = ax2.contour(x, y, Z, levels=20)
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
    contour3 = ax3.contour(x, y, Z, levels=20)
    ax3.set_xlabel('mu')
    ax3.set_ylabel('sigma')
    plt.colorbar(contour3, ax=ax3)
    for idx, trajectory in enumerate(Y_list):
        # Plot trajectories with mu on x-axis and sigma on y-axis
        ax3.plot(trajectory[2,:4000,0], trajectory[3,:4000,0], label=methods[idx], linewidth=1, color=colors[idx], alpha=alphas[idx])
    ax3.legend()


    # Plot contours of density
    contour4 = ax4.contour(x, y, Z, levels=20)
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
    args = parser.parse_args()
    main(M=args.M, h=args.h, T=args.T)

