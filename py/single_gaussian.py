"""
single particle with irreversible perturbation, Gaussian case
"""
from function import *


def main(dim, diagonal, dt):
    # Parameters
    x0 = np.array([5.0]*dim)  # Initial position
    repeat = 500
    burn_in = int(1/dt*10)  # Burn-in period
    n_steps = int(1/dt*1000)  # Number of steps
    step_interval = n_steps//100

    S = np.diag(diagonal)

    opt_J = get_opt_J(S)
    nopt_J = get_nopt_J(S)
    # rand_J = get_rand_J(S)

    J_list = {'noJ': np.zeros((dim, dim)), 'opt_J': opt_J, 'nopt_J': nopt_J}
    # J_list = {'noJ': np.zeros((2,2)), 'opt_J': opt_J, 'nopt_J': nopt_J, 'rand_J': rand_J}
    ground_truth = [0, np.trace(np.linalg.inv(S)), 0]
    av, bias2, vas = {}, {}, {}
    for name in J_list:
        J = J_list[name]
        _, av[name], bias2[name], vas[name] = get_bias_var_av(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, step_interval, scaled_gaussian)


    fig, ax = plt.subplots(3,3, dpi = 120, figsize=(12, 12))
    bins = step_interval * dt * np.arange(1, len(bias2['noJ'][0])+1)
    for i in range(3):
        for name in J_list:
            ax[0,i].loglog(bins, bias2[name][i], label = name)
            ax[1,i].loglog(bins, vas[name][i], label = name)
            ax[2,i].loglog(bins, bias2[name][i]+vas[name][i], label = name)
        ax[0,i].legend()
        ax[1,i].legend()
        ax[2,i].legend()
    ax[0,0].set_title('Bias^2 of E[x+y]')
    ax[0,1].set_title('Bias^2 of E[x^2+y^2]')
    ax[0,2].set_title('Bias^2 of E[x*y]')
    ax[1,0].set_title('Var of E[x+y]')
    ax[1,1].set_title('Var of E[x^2+y^2]')
    ax[1,2].set_title('Var of E[x*y]')
    ax[2,0].set_title('MSE^2 of E[x+y]')
    ax[2,1].set_title('MSE^2 of E[x^2+y^2]')
    ax[2,2].set_title('MSE^2 of E[x*y]')
    plt.suptitle('dim = ' + str(dim) + ', diag = ' + str(diagonal) +  ', dt = ' + str(dt))
    plt.savefig('plottings/{:d}-D Gaussian, dt={:.3f}'.format(dim, dt) + '.png', format='png', dpi=200)


if __name__ == "__main__":
    dimension = sys.argv[1]
    diagonal = sys.argv[2]
    dt = sys.argv[3]

    main(dim=int(dimension), 
         diagonal=np.fromstring(diagonal, sep=','), 
         dt=float(dt))

    # main(dim=2, 
    #      diagonal=[1,1/20], 
    #      dt=0.01)