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



import numpy as np
from numpy.linalg import slogdet, inv
from scipy.linalg import sqrtm

# ------------------------------------------------------------------
# 1.  Log–posterior   log π(W | X)   (up to a constant)
# ------------------------------------------------------------------
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
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 2.  One Metropolised Irreversible Riemannian Langevin step
# ------------------------------------------------------------------
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
# ------------------------------------------------------------------
