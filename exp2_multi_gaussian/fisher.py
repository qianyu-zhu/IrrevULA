import numpy as np
from scipy.stats import multivariate_normal


def score_log_gmm(x, means, weights, covs, inv_covs):
    K = len(means)
    D = len(x)
    scores = np.zeros((K, D))
    pdf_vals = np.zeros(K)

    for k in range(K):
        diff = x - means[k]
        inv_cov = inv_covs[k]
        pdf = multivariate_normal.pdf(x, mean=means[k], cov=covs[k])
        scores[k] = -inv_cov @ diff  # score of log N(x | mu_k, cov_k)
        pdf_vals[k] = weights[k] * pdf

    pi_x = np.sum(pdf_vals) + 1e-12
    weighted_scores = np.sum((pdf_vals[:, None] * scores), axis=0) / pi_x
    return weighted_scores  # ∇ log π(x)

def fisher_info_pi(means, weights, covs, n_samples=50000):
    D = len(means[0])
    inv_covs = [np.linalg.inv(cov) for cov in covs]
    K = len(means)

    samples = []
    for _ in range(n_samples):
        k = np.random.choice(K, p=weights)
        x = np.random.multivariate_normal(means[k], covs[k])
        samples.append(x)

    samples = np.array(samples)
    fisher = np.zeros((D, D))

    for x in samples:
        grad_log_pi = score_log_gmm(x, means, weights, covs, inv_covs)
        fisher += np.outer(grad_log_pi, grad_log_pi)

    fisher /= n_samples
    return fisher