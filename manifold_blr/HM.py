import numpy as np
import scipy.io as sio

def log_posterior(w, X, t, alpha):
    """
    Log of unnormalized posterior for Bayesian logistic regression.
    Prior: N(0, alpha^{-1} I)
    Likelihood: logistic
    """
    t = t.ravel()
    Xw = X @ w
    log_prior = -0.5 * alpha * np.dot(w, w)
    log_likelihood = np.dot(t, Xw) - np.sum(np.log1p(np.exp(Xw)))
    return log_prior + log_likelihood

def metropolis_hastings_sampler(X, t, alpha=1.0, Nsteps=250000, prop_std=0.15, burn_in=10000, seed=0):
    np.random.seed(seed)
    d = X.shape[1]
    w_now = np.zeros(d)
    chain = np.zeros((Nsteps, d))
    accepted = 0

    for i in range(Nsteps):
        w_prop = w_now + prop_std * np.random.randn(d)
        log_ratio = log_posterior(w_prop, X, t, alpha) - log_posterior(w_now, X, t, alpha)
        if np.log(np.random.rand()) < log_ratio:
            w_now = w_prop
            accepted += 1
        chain[i] = w_now

    print(f"Acceptance rate: {accepted / Nsteps:.4f}")
    return chain[burn_in:]  # Discard burn-in


def stat_func(Y):
    # Y: (d, K//N)
    # we compute some statistics of the chain
    stats = np.array([np.sum(np.abs(Y), axis=1), np.sum(Y**2, axis=1), \
                     np.max(np.abs(Y), axis=1), np.max(Y, axis=1), \
                     (Y[:, 3] > 0), (Y[:, 3] > 1), \
                     (Y[:, 3] > 2), (Y[:, 3] > 3), (Y[:, 3] > 4), \
                     (Y[:, 0] > -1) & (Y[:, 1] > 0), (Y[:, 3] > 3) & (Y[:, 4] > 1), \
                     np.sum(Y, axis=1), Y[:, 1] * Y[:, 4]])
    return np.mean(stats, axis=1) #(11,)


statistics_name = ['|X[4]|',r'$X[4]^2$', 'P(X[4]>0)', 'P(X[4]>1)', 'P(X[4]>2)', 'P(X[4]>3)', 'P(X[4]>4)', 'P(X[0,1]>0)', 'P(X[3,4]>0)', 'sum(X)', 'X[1]*X[4]']

# def stat_func(Y):
#     # Y: (Nsteps, d)
#     # we compute some statistics of the chain
#     stats = np.array([np.sum(np.abs(Y), axis=1), np.sum(Y**2, axis=1), \
#                      np.max(np.abs(Y), axis=1), np.max(Y, axis=1), \
#                      (Y[:,0] > 0) & (Y[:,1] > 0).astype(int), (Y[:,3] > 0) & (Y[:,4] > 0).astype(int), \
#                      np.sum(Y, axis=1), np.exp(np.sum(np.abs(Y), axis=1)/2)]) #(8, Nsteps)
#     return np.mean(stats, axis=1) #(8,)


# Load data
data = sio.loadmat('benchmarks.mat')
set_ = data['german'][0, 0]
train = set_['test']
test = set_['train']
ind = 42
xtrain = set_['x'][train[:, ind].astype(int), :]
ttrain = set_['t'][train[:, ind].astype(int), :]
xtest = set_['x'][test[:, ind].astype(int), :]
ttest = set_['t'][test[:, ind].astype(int), :]
ttrain = (ttrain == 1).astype(int)
ttest = (ttest == 1).astype(int)

# Run sampler
stats_list = []
for ii in range(20):
    w_samples = metropolis_hastings_sampler(xtrain, ttrain, alpha=1, Nsteps=250000, prop_std=0.15, burn_in=10000, seed=ii) #(Nsteps, d)
    stats = stat_func(w_samples) #(8,)
    stats_list.append(stats) #(20, 8)

stats_mean = np.mean(stats_list, axis=0) #(8,)
stats_std = np.std(stats_list, axis=0) #(8,)
print('mean of stats: ', stats_mean)
print('std of stats: ', stats_std)


to_save = {'stats_mean': stats_mean, 'stats_std': stats_std}
np.save('stats_list.npy', to_save)

