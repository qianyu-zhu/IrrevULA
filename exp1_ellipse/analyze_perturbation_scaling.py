"""
single particle with irreversible perturbation, Gaussian case
compare the performance of different J under the spectral-optimal constraint
"""
import os
import sys
import copy
import scipy
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from scipy.stats import multivariate_normal
from helper import getoptJ, get_bias_var_av, scaled_gaussian
plt.rcParams['figure.dpi'] = 120

    
colors = [
    "#92c5de",
    "#4393c3",
    "#d6604d",
    "#f4a582",
    "#b2182b",
    "#2166ac"
]

styles = [
    {"style": "--", "marker": "o"},
    {"style": "--", "marker": "s"},
    {"style": "--", "marker": "^"},
    {"style": "--", "marker": "v"},
    {"style": "--", "marker": "*"},
    {"style": "--", "marker": "D"},
]

def main(dim, diagonal, dt):
    # Parameters
    x0 = np.array([5.0]*dim)  # Initial position
    repeat = 5
    burn_in = int(1/dt*10)  # Burn-in period
    n_steps = int(1/dt*5000)  # Number of steps
    step_interval = n_steps//100


    k = 5
    S = np.diag(diagonal)
    opt_J = getoptJ(S)
    J_list = [opt_J * i * 5 for i in range(k)]

    ground_truth = [0, np.trace(np.linalg.inv(S)), 0]
    
    av, bias2, vas = {}, {}, {}
    title = [fr'MSE$^2$ of $\mathbb{{E}}[x+y]$', fr'MSE$^2$ of $\mathbb{{E}}[x^2+y^2]$', fr'MSE$^2$ of $\mathbb{{E}}[x*y]$']
    for i in range(0, k):
        J = J_list[i]
        _, av[i], bias2[i], vas[i] = get_bias_var_av(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, step_interval, scaled_gaussian)


    fig, ax = plt.subplots(1, 3, dpi = 200, figsize=(12, 4))
    bins = step_interval * dt * np.arange(1, len(bias2[0][0])+1)
    for j in range(3):
        for i in range(0, k):
            multiplier = i
            label = r'$J = J^* \times {:.0f}$'.format(multiplier)
            ax[j].loglog(bins, bias2[i][j]+vas[i][j], 
                        label=label, 
                        color=colors[i], 
                        linestyle=styles[i]['style'], 
                        marker=styles[i]['marker'])
        ax[j].legend()
        ax[j].grid(True, which='major', linestyle='-')
        ax[j].grid(True, which='minor', linestyle='--', alpha=0.2)
        ax[j].set_title(title[j])

    # Ensure output directory exists before saving
    os.makedirs('plottings', exist_ok=True)
    plt.suptitle('dim = ' + str(dim) + ', diag = ' + str(diagonal) +  ', dt = ' + str(dt), y=1.1)
    plt.savefig('compare_J_{:d}-D Gaussian, diag={}, dt={:.3f}'.format(dim, str(diagonal)[1:-1], dt) + '.png', format='png', dpi=200)


if __name__ == "__main__":
    dimension = 4
    diagonal = [2**i for i in range(dimension)]
    dt = 0.04

    main(dim=int(dimension), 
         diagonal=np.array(diagonal), 
         dt=float(dt))