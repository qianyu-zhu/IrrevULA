import sys
import copy
import scipy
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from scipy.stats import multivariate_normal
plt.rcParams['figure.dpi'] = 120


def gram_schmidt_process(A):
    n = A.shape[1]
    dim = A.shape[0]
    # Initialize the orthonormal basis matrix
    Q = np.zeros((dim, n)).astype('float')
    
    for i in range(n):
        # Start with the original vector
        q = A[:, i]
        for j in range(i):
            # Subtract the projection of q onto the j-th orthonormal vector
            q -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
        # Normalize the vector
        Q[:, i] = q / np.linalg.norm(q)
    
    return Q #notice that the Q is the eigenvector matrix (column matrix)


def construct_Q(S):
    d = len(S)
    Q = np.zeros(S.shape)
    gamma = np.trace(S)/d
    # basis = np.eye(d)
    basis = ortho_group.rvs(dim=d)

    # construct the i-th column of Q
    for i in range(d-1):
        inner_prod_S = np.array([np.real(np.dot(basis[:,i], np.dot(S, basis[:,i]))) for i in range(basis.shape[1])])
        equal_index = next((index for index, value in enumerate(inner_prod_S) if value == gamma), -1)
        # print(equal_index)
        if equal_index != -1:
            Q[:,i] = basis[:,equal_index]
            basis = np.delete(basis, equal_index, axis=1)
            basis = gram_schmidt_process(basis)
            continue
        big_index = np.argmax(inner_prod_S)
        small_index = np.argmin(inner_prod_S)
        # big_index = next((index for index, value in enumerate(inner_prod_S) if value > gamma), -1)
        # small_index = next((index for index, value in enumerate(inner_prod_S) if value < gamma), -1)
        big_vec = basis[:,big_index]
        small_vec = basis[:,small_index]
        # print('inner_prod_S, big_vec, small_vec', inner_prod_S, big_vec, small_vec)
        a0, a1, b = inner_prod_S[small_index], inner_prod_S[big_index], np.dot(basis[:,big_index], np.dot(S, basis[:,small_index]))
        tan_theta = (-b+np.sqrt(b**2+(a1-gamma)*(gamma-a0)))/(a1-gamma)
        # print('a0, a1, b, tan_theta', a0, a1, b, tan_theta)
        new_vec = (small_vec + tan_theta*big_vec)/np.sqrt(1+tan_theta**2)
        # print('new_vec inner prod:', new_vec, new_vec@S@new_vec)
        Q[:,i] = new_vec
        basis = np.delete(basis, big_index, axis=1)
        new_basis = np.insert(basis, 0, new_vec, axis=1)
        basis = gram_schmidt_process(new_basis)[:,1:]
    Q[:,-1] = basis[:,0]

    return Q

# output is J, not J_tilde!!!
def construct_J(Q, S, Lambda, random_Lambda=False):
    d = len(Q)
    J = np.zeros(Q.shape)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if random_Lambda:
                J[i,j] = (Lambda[i]+Lambda[j])/(Lambda[i]-Lambda[j]) * np.dot(Q[:,i], np.dot(S, Q[:,j]))
            else:
                J[i,j] = np.sign(Lambda[i]-Lambda[j]) * np.dot(Q[:,i], np.dot(S, Q[:,j]))
    J_tilde = Q@J@Q.T
    sqrt_S_inv = np.linalg.inv(scipy.linalg.sqrtm(S))
    J = sqrt_S_inv@J_tilde@sqrt_S_inv
    return J



def langevin_dynamics_high_dim(x0, potential_grad, dt, n_steps, S, J = None):
    dim = len(x0)
    if J is None:
        J = np.diag(np.zeros(dim))

    # Initialize position and trajectory
    x = copy.copy(x0)
    trajectory = [x]
    dim = len(x0)
    sqrt_2dt = np.sqrt(2 * dt)
    
    # Langevin dynamics loop
    for _ in range(n_steps):
        # Compute deterministic force
        force = -potential_grad(x, J, S)
        # print(force)
        
        # Generate random noise
        noise = np.random.normal(size=dim)
        
        # Langevin equationiven
        x = x + dt * force + sqrt_2dt * noise
        trajectory.append(x)
    
    return np.array(trajectory)

# S = ...
def scaled_gaussian(x, J, S):
    dim = len(x)
    sigma = (np.eye(dim)+J)@S
    return sigma@x


def get_opt_J(S):
    d = S.shape[0]
    Q = construct_Q(S)
    # Lambda = np.arange(1,d+1)
    Lambda = np.random.rand(d)

    J = construct_J(Q, S, Lambda, random_Lambda=False)
    return J

def get_nopt_J(S):
    d = S.shape[0]
    Q = construct_Q(S)
    # Lambda = np.arange(1,d+1)
    Lambda = np.random.rand(d)

    J = construct_J(Q, S, Lambda, random_Lambda=True)
    return J

def generate_skew_symmetric_matrix(n):
    A = np.random.rand(n, n)
    return A - A.T

def get_rand_J(S):
    return generate_skew_symmetric_matrix(S.shape[0])

def get_statistics(step_interval, func):
    moving_mean = []
    for i in range(step_interval, len(func), step_interval):
        moving_mean.append(np.mean( func[:i] ))
    return np.array(moving_mean)

def asymptotic_variance(func, num_split = 20):
    split_means = []
    split_func = np.array_split(func, num_split)
    for i in split_func:
        split_means.append(np.mean( i ))
    return np.var(split_means)

def get_estimators(trajectory, burn_in, S, ground_truth, step_interval = 1000):
    trajectory_burned = trajectory[burn_in:]
    func_1 = np.sum(trajectory_burned, axis = 1) - ground_truth[0] # x+y
    func_2 = np.linalg.norm(trajectory_burned, axis = 1)**2 - ground_truth[1] # x**2 + y**2 - 11
    func_3 = np.prod(trajectory_burned, axis = 1) - ground_truth[2] # x*y
    func_list = [func_1, func_2, func_3]
    estimator = [get_statistics(step_interval, func) for func in func_list]
    av = [asymptotic_variance(func, num_split = 20) for func in func_list]
    return np.array(estimator), np.array(av)

def get_bias_var_av(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, step_interval, gradient):
    stats = {} # disc: 
    avs = [] # 1 for each statistic, 3 in total
    bias2 = [] # 3
    var = [] # 3
    for _ in tqdm(range(repeat)):
        trajectory = langevin_dynamics_high_dim(x0, gradient, dt, n_steps, S, J)
        estimators, av = get_estimators(trajectory, burn_in, S, ground_truth, step_interval)
        for i in range(3):
            if i not in stats:
                stats[i] = []
            stats[i].append(estimators[i])
        avs.append(av)
    for i in range(3):
        bias2.append(np.mean(np.array(stats[i]), axis=0)**2)
        var.append(np.var(np.array(stats[i]), axis=0))
    avs = np.mean(np.array(avs), axis=1)
    return stats, avs, bias2, var

def get_var_MSE(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, gradient):
    MSE2 = []
    for _ in range(repeat):
        trajectory = langevin_dynamics_high_dim(x0, gradient, dt, n_steps, S, J)
        trajectory_burned = trajectory[burn_in:]
        func_2 = np.linalg.norm(trajectory_burned, axis = 1)**2 - ground_truth[1] # x**2 + y**2 - 11
        MSE2.append(np.mean(func_2)**2)
    return np.mean(MSE2)

def get_bias_var_av_sum(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, gradient):
    MSE2 = []
    for _ in range(repeat):
        trajectory = langevin_dynamics_high_dim(x0, gradient, dt, n_steps, S, J)
        trajectory_burned = trajectory[burn_in:]
        func_1 = np.sum(trajectory_burned, axis = 1) - ground_truth[0] # x+y
        func_2 = np.linalg.norm(trajectory_burned, axis = 1)**2 - ground_truth[1] # x**2 + y**2 - 11
        func_3 = np.prod(trajectory_burned, axis = 1) - ground_truth[2] # x*y
        MSE2.append(np.sum([np.mean(func)**2 for func in [func_1, func_2, func_3]]))
    return np.mean(MSE2) # (1, )


def get_bias_var_av_sum_separate(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, gradient):
    MSE2 = []
    for _ in range(repeat):
        trajectory = langevin_dynamics_high_dim(x0, gradient, dt, n_steps, S, J)
        trajectory_burned = trajectory[burn_in:]
        func_1 = np.sum(trajectory_burned, axis = 1) - ground_truth[0] # x+y
        func_2 = np.linalg.norm(trajectory_burned, axis = 1)**2 - ground_truth[1] # x**2 + y**2 - 11
        func_3 = np.prod(trajectory_burned, axis = 1) - ground_truth[2] # x*y
        MSE2.append([np.mean(func)**2 for func in [func_1, func_2, func_3]]) # (repeat, 3)
    return np.mean(MSE2, axis=0) # (3, )

def get_MMD(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, gradient, n_samples):
    MSE2 = []
    for _ in range(repeat):
        trajectory = langevin_dynamics_high_dim(x0, gradient, dt, n_steps, S, J)
        trajectory_burned = trajectory[burn_in:]
        target_sample = np.random.multivariate_normal(np.zeros(len(x0)), np.linalg.inv(S), size=n_samples)
        
        # Compute pairwise distances within and between samples
        K_xx = np.sum(np.square(trajectory_burned[:, None] - trajectory_burned), axis=2)
        K_yy = np.sum(np.square(target_sample[:, None] - target_sample), axis=2) 
        K_xy = np.sum(np.square(trajectory_burned[:, None] - target_sample), axis=2)
        
        # Compute MMD using Gaussian kernel with median heuristic
        # Z = np.vstack((trajectory_burned, target_sample))
        # pairwise_dists = np.sum((Z[:, np.newaxis] - Z[np.newaxis, :]) ** 2, axis=2)
        # sigma = np.median(pairwise_dists)
        sigma = np.median(K_xx)
        K_xx = np.exp(-K_xx / (2 * sigma))
        K_yy = np.exp(-K_yy / (2 * sigma))
        K_xy = np.exp(-K_xy / (2 * sigma))
        
        mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
        MSE2.append(mmd)
    return np.mean(MSE2) # (1, )