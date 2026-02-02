import copy
import scipy
import numpy as np
from scipy.stats import ortho_group
from numpy.linalg import slogdet, inv
from scipy.linalg import sqrtm

"""helper functions for constructing the perturbations J"""
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
    for _ in range(100):
        Q = construct_Q(S)
        Lambda = np.random.rand(d)
        J_current = construct_J(Q, S, Lambda, random_Lambda=False)
        if np.trace(-J_current@S@J_current) < value:
            J = J_current
            value = np.trace(-J_current@S@J_current)
    return J

def getnoptJ(S):
    d = S.shape[0]
    J_opt = getoptJ(S)
    value = np.linalg.norm(J_opt, ord='fro')
    print('J_opt', J_opt)
    print('value', value)
    for i in range(10000):
        Q = construct_Q(S)
        Lambda = np.random.rand(d)
        J = construct_J(Q, S, Lambda, random_Lambda=True)
        if np.linalg.norm(J, ord='fro') < 5*value:
            return J
    print('no J found')
    return J

def getJ(S):
    d = S.shape[0]
    J = np.random.rand(d,d)
    J = (J - J.T)/2
    return J/np.linalg.norm(J, ord='fro')


### Functions for ICA
def gradlogpos(W, X, lambda_):
    d, N = X.shape
    n = int(0.1 * N)
    summat = np.zeros((d, d))
    W = W.reshape(d, d)

    index = np.random.choice(N, n, replace=False)
    for idx in index:
        xn = X[:, idx]
        yn = W @ xn
        summat += np.tanh(0.5 * yn)[:, np.newaxis] @ xn[np.newaxis, :]

    dlogpi = (N * np.linalg.inv(W.T) - (N / n) * summat) - lambda_ * W
    return dlogpi.flatten()


def gradlogpos_irrwriem_new(W_vec, X, J, lambd=1):
    """
    Compute gradient of log-posterior with irreversible + Riemannian metric terms.

    Parameters:
        W_vec: (d*d,) numpy array, flattened matrix W
        X:     (d, N) data matrix
        J:     (d*d, d*d) skew-symmetric matrix (irreversible perturbation)
        lambd: float, regularization parameter

    Returns:
        dlogpi: (d*d,) numpy array, flattened gradient
    """
    d, N = X.shape
    n = round(0.1 * N)

    W = W_vec.reshape(d, d)

    summat1 = np.zeros((d, d))
    summat2 = np.zeros((d, d))

    # Random sample n indices from N without replacement
    index = np.random.choice(N, size=n, replace=False)
    for ii in index:
        xn = X[:, ii]
        yn = W @ xn
        t = np.tanh(0.5 * yn)
        summat1 += np.outer(t, yn)
        summat2 += np.outer(t, xn)
    W_T_inv = np.linalg.pinv(W.T)

    dlogpi1 = (N * np.eye(d) - (N / n) * summat1) @ W - lambd * W @ (W.T @ W) + (d + 1) * W
    dlogpi2 = (N * np.eye(d) @ W_T_inv - (N / n) * summat2) - lambd * W
    dlogpi = dlogpi1.flatten() + dlogpi2.flatten() + J @ dlogpi2.flatten()
    return dlogpi


def gradlogpos_riem_new(W_vec, X, lam):
    """
    Translated from MATLAB function gradlogpos_riem_new.

    Parameters:
    - W_vec: flattened (d*d,) numpy array
    - X: (d, N) array of samples
    - lam: scalar lambda

    Returns:
    - dlogpi: flattened (d*d,) numpy array
    """
    d, N = X.shape
    n = round(0.1 * N)

    summat1 = np.zeros((d, d))
    summat2 = np.zeros((d, d))
    W = W_vec.reshape((d, d))

    index = np.random.choice(N, size=n, replace=False)

    for ii in index:
        xn = X[:, ii]
        yn = W @ xn
        tyn = np.tanh(0.5 * yn)
        summat1 += np.outer(tyn, yn)
        summat2 += np.outer(tyn, xn)

    # Compute dlogpi1
    dlogpi1 = (N * np.eye(d) - (N / n) * summat1) @ W \
              - lam * W @ (W.T @ W) + (d + 1) * W

    # Compute dlogpi2
    dlogpi2 = np.linalg.solve(W.T, N * np.eye(d)) - (N / n) * summat2 - lam * W

    # Combine and flatten
    dlogpi = dlogpi1.ravel() + dlogpi2.ravel()

    return dlogpi


def logpi(W_vec, X, lambd=1.0):
    """
    Log-posterior for the ICA model used by Welling & Teh (2011).

    log π(W|X) = log|det W|
               + Σ_{n,i} 2 log sech( ½ w_i^T x_n )
               – (λ/2) ||W||_F²               + const
    """
    d, N = X.shape
    W = W_vec.reshape(d, d)

    # log|det W|
    sign, logdet = slogdet(W)
    if sign <= 0:                        # singular or negative det → -∞
        return -np.inf

    # likelihood term   p(y) ∝ sech²(½y)
    Y = W @ X                            # shape (d, N)
    loglik = 2.0 * np.sum(-np.log(2*np.cosh(0.5*Y)))   # drop −log4 constant

    # Gaussian prior N(0, λ⁻¹)
    logprior = -0.5 * lambd * np.sum(W**2)

    return logdet + loglik + logprior    # constant term ignored


def irmala_step(W_vec, X, J, lambd, dt):
    """
    One Metropolis-adjusted step of irreversible Riemannian Langevin
    (metric  G = I + WᵀW, irreversible drift  J∇logπ).

    Returns:
        W_new  – accepted or current position
        accept – boolean flag
    """
    d, N = X.shape
    D = d**2

    # ---------- helper functions ----------
    def metric(Wm):                          # (d,d) → (d,d)
        return np.eye(d) + Wm.T @ Wm

    def sqrt_metric(Wm):
        return sqrtm(metric(Wm)).real        # real part, numerical guard

    # ---------- current quantities ----------
    W  = W_vec
    Wm = W.reshape(d, d)
    grad = gradlogpos_irrwriem_new(W, X, J, lambd)
    G   = metric(Wm)
    S   = sqrt_metric(Wm)                    # G^{1/2}

    # mean and covariance of forward proposal
    mu_fwd   = W + 0.5*dt * grad
    Sigma_f  = dt * np.kron(G, np.eye(d))    # (D,D) covariance

    # draw proposal
    eps      = np.random.randn(d, d)
    noise    = eps @ S                       # (d,d)  ~ N(0, G)
    W_prop   = mu_fwd + np.sqrt(dt) * noise.reshape(D)

    # ---------- proposal densities ----------
    # backward quantities
    Wpm   = W_prop.reshape(d, d)
    G_p   = metric(Wpm)
    S_p   = sqrt_metric(Wpm)
    grad_p = gradlogpos_irrwriem_new(W_prop, X, J, lambd)
    mu_bwd = W_prop + 0.5*dt * grad_p
    Sigma_b = dt * np.kron(G_p, np.eye(d))

    # log-Gaussian helper
    def log_gauss(x, mean, cov_chol):        # cov_chol = Σ^{1/2}
        diff = x - mean
        # solve Σ^{-1/2} diff without forming Σ
        sol  = np.linalg.solve(cov_chol, diff.reshape(d, d)).ravel()
        quad = 0.5 * sol @ sol
        logdet = np.sum(np.log(np.diag(cov_chol))) * 2
        return -quad - 0.5*D*np.log(2*np.pi) - 0.5*logdet

    log_q_fwd = log_gauss(W_prop, mu_fwd, np.sqrt(dt)*np.kron(S, np.eye(d)))
    log_q_bwd = log_gauss(W,      mu_bwd, np.sqrt(dt)*np.kron(S_p, np.eye(d)))

    # ---------- Metropolis ratio ----------
    log_alpha = ( logpi(W_prop, X, lambd) - logpi(W, X, lambd)
                + log_q_bwd               - log_q_fwd )
    if np.log(np.random.rand()) < log_alpha:
        return W_prop, True
    else:
        return W, False


def log_gaussian(x, mean, cov):
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    log_det = np.linalg.slogdet(cov)[1]
    return -0.5 * diff @ cov_inv @ diff - 0.5 * log_det - 0.5 * len(x) * np.log(2 * np.pi)

def compute_log_alpha(W, W_prop, grad_W, grad_W_prop, dt):
    # Reshape
    d = int(np.sqrt(len(W)))
    W_mat = W.reshape(d, d)
    Wp_mat = W_prop.reshape(d, d)

    # Metric
    G = np.eye(d) + W_mat.T @ W_mat
    Gp = np.eye(d) + Wp_mat.T @ Wp_mat

    G_full = dt * np.kron(G, np.eye(d))
    Gp_full = dt * np.kron(Gp, np.eye(d))

    mu = W + 0.5 * dt * grad_W
    mup = W_prop + 0.5 * dt * grad_W_prop

    # Note: logpi requires X and lambd parameters, this function may need to be updated
    # logpi_ratio = logpi(W_prop, X, lambd) - logpi(W, X, lambd)
    logpi_ratio = 0  # Placeholder - this function needs X and lambd to work properly
    q_forward = log_gaussian(W_prop, mu, G_full)
    q_backward = log_gaussian(W, mup, Gp_full)

    return logpi_ratio + q_backward - q_forward
