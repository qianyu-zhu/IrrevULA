import numpy as np

def grad_logpos_blr(w, alpha, X, t, n):
    return -alpha * w + log_likelihood_blr(w, t, X, n)

def grad_logpos_blr_irriem(w, alpha, X, t, n, G, J):
    output = -alpha * w + log_likelihood_blr(w, t, X, n)
    
    # Compute the expression step by step
    output = output + 2 * np.dot(J, output)
    output += np.dot((np.eye(len(w)) + J), np.linalg.solve(G, output))
    output += np.linalg.solve(G, np.dot(J, output))
    output -= correction2(X, w, G, J)

    return output


def grad_logpos_blr_irwriem(w, alpha, X, t, n, G, J):
    output = -alpha * w + log_likelihood_blr(w, t, X, n)
    
    # Compute the expression step by step
    output = np.linalg.solve(G, output) + output - correction(X, w, G) + np.dot(J, output)
    
    return output

def grad_logpos_blr_riem(w, alpha, X, t, n, G):
    # Compute initial part of the gradient
    output = -alpha * w + log_likelihood_blr(w, t, X, n)
    
    # Apply Riemannian gradient transformation
    output = np.linalg.solve(G, output) + output - correction(X, w, G)
    
    return output


def logistic_func(a):
    return 1 / (1 + np.exp(-a))


import numpy as np

def log_likelihood_blr(w, t, X, n):
    N = X.shape[1]

    if n < N:
        index = np.random.choice(N, n, replace=False)
        Xnow = X[:, index]
        tnow = t[index].ravel()
    else:
        Xnow = X
        tnow = t

    phi = lambda y: 1 / (1 + np.exp(-y))
    gradl = (N / n) * (Xnow @ tnow - Xnow @ phi(Xnow.T @ w))

    return gradl

def correction(X, w, G):
    d = w.shape[0]
    
    logiteval = logistic_func(np.dot(X.T, w))
    Lambda = np.diag(logiteval * (1 - logiteval))
    
    output = np.zeros((d, 1))
    
    for ii in range(d):
        Vii = np.diag((1 - 2 * logistic_func(np.dot(X.T, w))) * X[ii, :])
        dG = np.dot(np.dot(X, Lambda), np.dot(Vii, X.T))
        termii = np.linalg.solve(G, dG) @ np.linalg.inv(G)
        output += termii[:, ii].reshape(-1, 1)
    
    return output


def correction2(X, w, G, J):
    d = w.shape[0]
    
    logiteval = logistic_func(np.dot(X.T, w))
    Lambda = np.diag(logiteval * (1 - logiteval))
    
    output = np.zeros((d, 1))
    
    for ii in range(d):
        Vii = np.diag((1 - 2 * logistic_func(np.dot(X.T, w))) * X[ii, :])
        dG = np.dot(np.dot(X, Lambda), np.dot(Vii, X.T))
        termii = np.linalg.solve(G, dG) @ np.linalg.inv(G)
        termii = termii + J @ termii + termii @ J
        output += termii[:, ii].reshape(-1, 1)
    
    return output

def Logpos(w, alpha, X, ttrain):
    return -alpha * np.linalg.norm(w) ** 2 / 2 + np.dot(ttrain, np.dot(X.T, w)) - np.sum(np.log(1 + np.exp(np.dot(X.T, w))))



def stat_func(Y):
    # Y: (d, K//N)
    # we compute some statistics of the chain
    return np.array([np.mean(np.sum(np.abs(Y), axis=0)), np.mean(np.sum(Y**2, axis=0)), \
                     np.mean(np.max(np.abs(Y), axis=0)), np.mean(np.max(Y, axis=0)), \
                     np.mean((Y[3, :] > 0)), np.mean((Y[3, :] > 1)), \
                     np.mean((Y[3, :] > 2)), np.mean((Y[3, :] > 3)), np.mean((Y[3, :] > 4)), \
                     np.mean((Y[0, :] > -1) & (Y[1, :] > 0)), np.mean((Y[3, :] > 3) & (Y[4, :] > 1)), \
                     np.mean(np.sum(Y, axis=0)), np.mean(Y[1,:] * Y[4, :])]) #(11,)






