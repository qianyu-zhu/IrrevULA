import numpy as np
"""helper functions for adaptive Fisher information matrix and its inverse"""
def initialize_A1(s1, lam=1):
    """Initialize A1 = (s1 s1^T + λ I_d)^(-1)"""
    d = s1.shape[0]
    outer = np.outer(s1, s1)
    A1 = (1 / lam) * (np.eye(d) - outer / (lam + s1 @ s1))
    return A1

def iterate_An(An_minus_1, sn, i):
    """Update A_n using the recursive formula"""
    An_minus_1 = An_minus_1 * (i+1) / i
    sn = sn / np.sqrt(i+1)
    numerator = An_minus_1 @ np.outer(sn, sn) @ An_minus_1
    denominator = 1 + sn.T @ An_minus_1 @ sn
    An = An_minus_1 - numerator / denominator
    return An

def initialize_FIM(sn, K = 500):
    return np.eye(4)

def iterate_FIM(FIM, sn, i, K = 500):
    eta = 1.0 / (K + i)
    return (1.0 - eta) * FIM + eta * np.outer(sn, sn)