import numpy as np

def gradfunc(state, data):
    """
    Computes the gradient for the given state and data.
    
    Parameters:
    state: List or array with [mu, sigma]
    data: Array of data points (N, 2)
    
    Returns:
    grad: Numpy array of gradients [grad_mu, grad_sigma]
    """
    mu_1, sigma_1, mu_2, sigma_2 = state
    data_1, data_2 = data[:, 0], data[:, 1]
    N = len(data)
    
    # Compute moments
    m1 = np.sum(data_1 - mu_1)
    m2 = np.sum((data_1 - mu_1) ** 2)
    m3 = np.sum(data_2 - mu_2)
    m4 = np.sum((data_2 - mu_2) ** 2)

    # Initialize the gradient
    grad = np.zeros(4)
    
    # Gradient calculations
    grad[0] = m1 / (sigma_1 ** 2)  # Gradient for mu
    grad[1] = -N / sigma_1 + m2 / (sigma_1 ** 3)  # Gradient for sigma
    grad[2] = m3 / (sigma_2 ** 2)  # Gradient for mu
    grad[3] = -N / sigma_2 + m4 / (sigma_2 ** 3)  # Gradient for sigma
    
    return grad
