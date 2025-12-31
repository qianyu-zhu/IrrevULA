import numpy as np

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

    logpi_ratio = log_post(W_prop) - log_post(W)
    q_forward = log_gaussian(W_prop, mu, G_full)
    q_backward = log_gaussian(W, mup, Gp_full)

    return logpi_ratio + q_backward - q_forward