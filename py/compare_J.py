"""
single particle with irreversible perturbation, Gaussian case
compare the performance of different J under the spectral-optimal constraint
"""
from function import *

def main(dim, diagonal, dt):
    # Parameters
    x0 = np.array([5.0]*dim)  # Initial position
    repeat = 300
    burn_in = int(1/dt*10)  # Burn-in period
    n_steps = int(1/dt*5000)  # Number of steps
    step_interval = n_steps//100


    k = 5
    S = np.diag(diagonal)
    opt_J = get_opt_J(S)
    J_list = [opt_J * i for i in range(k)]

    ground_truth = [0, np.trace(np.linalg.inv(S)), 0]
    
    av, bias2, vas = {}, {}, {}
    title = ['MSE^2 of E[x+y]', 'MSE^2 of E[x^2+y^2]', 'MSE^2 of E[x*y]']
    for i in range(0, k-1):
        J = J_list[i]
        _, av[i], bias2[i], vas[i] = get_bias_var_av(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, step_interval, scaled_gaussian)

    fig, ax = plt.subplots(1, 3, dpi = 200, figsize=(12, 4))
    bins = step_interval * dt * np.arange(1, len(bias2[0][0])+1)
    for j in range(3):
        for i in range(0, k-1):
            ax[j].loglog(bins, bias2[i][j]+vas[i][j], label = str(i))
        ax[j].legend()
        ax[j].grid(True, which='major', linestyle='-')
        ax[j].grid(True, which='minor', linestyle='--', alpha=0.2)
        ax[j].set_title(title[j])

    plt.suptitle('dim = ' + str(dim) + ', diag = ' + str(diagonal) +  ', dt = ' + str(dt), y=1.1)
    plt.savefig('plottings/compare_J_{:d}-D Gaussian, diag={}, dt={:.3f}'.format(dim, str(diagonal)[1:-1], dt) + '.png', format='png', dpi=200)


if __name__ == "__main__":
    dimension = sys.argv[1]
    diagonal = sys.argv[2]
    dt = sys.argv[3]

    main(dim=int(dimension), 
         diagonal=np.fromstring(diagonal, sep=','), 
         dt=float(dt))