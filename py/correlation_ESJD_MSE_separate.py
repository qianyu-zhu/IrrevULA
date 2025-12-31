
"""
single particle with irreversible perturbation, Gaussian case
compare the performance of different J under the spectral-optimal constraint
"""
from function import *
import scipy.interpolate as si

def main(dim, diagonal, dt, fixed_evalues, repeat=20):

    # Parameters
    x0 = np.array([5.0]*dim)  # Initial position
    burn_in = int(1/dt*10)  # Burn-in period
    n_steps = int(1/dt*3000)  # Number of steps

    S = np.diag(diagonal)
    num_J = 300
    ground_truth = [0, np.trace(np.linalg.inv(S)), 0]

    trace_jsj, trace_jsjs, trace_jssj = [], [], []
    MSE2 = np.zeros((num_J, 3))

    for i in tqdm(range(num_J)):
        # non-optimal J: optimal in Lelivre's paper, 
        # but not optimal in the sense of ESJD
        if fixed_evalues == 1:
            J = get_opt_J(S)
        else:
            valid = False
            while not valid:
                J = get_nopt_J(S)
                # Check if all eigenvalues are less than 50
                valid = np.all(np.imag(np.linalg.eigvals(J))**2 < 4*1/diagonal[-1])
        # print(np.linalg.eigvals(J))
        trace_jsj.append(-np.trace(J@S@J))
        trace_jsjs.append(-np.trace(J@S@J@S))
        trace_jssj.append(-np.trace(J@S@S@J))
        MSE2[i] = get_bias_var_av_sum_separate(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, scaled_gaussian)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Convert lists to numpy arrays
    trace_jsj = np.array(trace_jsj)
    trace_jsjs = np.array(trace_jsjs)
    trace_jssj = np.array(trace_jssj)
    
    # Calculate number of entries to keep (95%)
    keep_count = int(0.95 * len(trace_jsj))
    
    # Get indices that would sort each metric
    jsj_sorted_indices = np.argsort(trace_jsj)
    jsjs_sorted_indices = np.argsort(trace_jsjs)
    jssj_sorted_indices = np.argsort(trace_jssj)
    
    # Get indices of entries to remove (top 5% from either metric)
    remove_jsj = set(jsj_sorted_indices[keep_count:])
    remove_jsjs = set(jsjs_sorted_indices[keep_count:])
    remove_jssj = set(jssj_sorted_indices[keep_count:])
    remove_indices = list(remove_jsj.union(remove_jsjs).union(remove_jssj))
    
    # Get indices to keep by creating a mask
    keep_mask = np.ones(len(trace_jsj), dtype=bool)
    keep_mask[remove_indices] = False
    
    # Filter arrays using the mask
    trace_jsj = trace_jsj[keep_mask]
    trace_jsjs = trace_jsjs[keep_mask]
    trace_jssj = trace_jssj[keep_mask]
    MSE2 = MSE2[keep_mask]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    titles = ['E[x+y]', 'E[x^2+y^2]', 'E[x*y]']

    for i in range(3):
        # Plot correlation JSJ and MSE
        axes[i,0].scatter(trace_jsj, MSE2[:,i], alpha=0.5)
        axes[i,0].set_xlabel('trace(-JSJ)', labelpad=10)
        axes[i,0].set_ylabel('MSE^2', labelpad=10)
        axes[i,0].set_title(f'Correlation between -tr(JSJ) and MSE of {titles[i]}', pad=10)
        axes[i,0].tick_params(axis='both', which='major', pad=8)

        # Plot correlation between JSJS and MSE 
        axes[i,1].scatter(trace_jsjs, MSE2[:,i], alpha=0.5)
        axes[i,1].set_xlabel('trace(-JSJS)', labelpad=10)
        axes[i,1].set_ylabel('MSE^2', labelpad=10)
        axes[i,1].set_title(f'Correlation between -tr(JSJS) and MSE of {titles[i]}', pad=10)
        axes[i,1].tick_params(axis='both', which='major', pad=8)

        # Plot correlation between JSSJ and MSE
        scatter = axes[i,2].scatter(trace_jsj, trace_jsjs, 
                                  c=MSE2[:,i], 
                                  cmap='viridis',
                                  alpha=0.5)
        plt.colorbar(scatter, ax=axes[i,2], label='MSE^2')
        axes[i,2].set_xlabel('trace(-JSJ)', labelpad=10)
        axes[i,2].set_ylabel('trace(-JSJS)', labelpad=10)
        axes[i,2].set_title('Correlation between -tr(JSJ) and -tr(JSJS)', pad=10)
        axes[i,2].tick_params(axis='both', which='major', pad=8)

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Set a single title above all subplots with extra spacing
    plt.suptitle(f'ESJD Correlation (point size ~ 1/MSE),\n diag=[{str(diagonal)[1:-1]}], dt={dt:.3f}, fixed_evalues={fixed_evalues}', y=1.02)

    plt.savefig('plottings/correlation_ESJD_MSE_{:d}-D Gaussian, diag=[{}], dt={:.3f}, fixed_evalues={}, repeat={}.png'.format(dim, str(diagonal)[1:-1], dt, fixed_evalues, repeat), format='png', dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    dimension = sys.argv[1]
    diagonal = sys.argv[2]
    dt = sys.argv[3]
    fixed_evalues = sys.argv[4]
    repeat = sys.argv[5]
    main(dim=int(dimension), 
         diagonal=np.fromstring(diagonal, sep=','), 
         dt=float(dt),
         fixed_evalues=int(fixed_evalues),
         repeat=int(repeat))
