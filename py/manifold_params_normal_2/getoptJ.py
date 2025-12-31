import copy
import scipy
import numpy as np
from scipy.stats import ortho_group


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


def getoptJ(S):
    d = S.shape[0]
    J = None
    value = np.inf
    for _ in range(1000):
        Q = construct_Q(S)
        Lambda = np.random.rand(d)
        J_current = construct_J(Q, S, Lambda, random_Lambda=False)
        if np.trace(-J_current@S@J_current) < value:
            J = J_current
            value = np.trace(-J_current@S@J_current)
    return J

def getnoptJ(S):
    d = S.shape[0]
    J = getoptJ(S)
    value = np.trace(-J@S@J)
    for _ in range(1000):
        Q = construct_Q(S)
        Lambda = np.random.rand(d)
        J = construct_J(Q, S, Lambda, random_Lambda=True)
        if np.trace(-J@S@J) > 2*value and np.trace(-J@S@J) < 3*value:
            return J
    return J

def getJ(S):
    d = S.shape[0]
    J = np.random.rand(d,d)
    J = (J - J.T)/2
    return J
# def getoptJ(S):
#     d = S.shape[0]
#     Q = construct_Q(S)
#     Lambda = np.arange(1,d+1)

#     J = construct_J(Q, S, Lambda, random_Lambda=False)
#     return J


# def getnoptJ(S):
#     d = S.shape[0]
#     Q = construct_Q(S)
#     Lambda = np.arange(1,d+1)

#     J = construct_J(Q, S, Lambda, random_Lambda=True)
#     return J