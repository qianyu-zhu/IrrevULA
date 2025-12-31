
"""
single particle with irreversible perturbation, Gaussian case
compare the performance of different J under the spectral-optimal constraint
"""
from function import *
import scipy.interpolate as si

def main(dim, diagonal, dt, fixed_evalues, repeat,n_samples):

    # Parameters
    x0 = np.array([0.0]*dim)  # Initial position
    burn_in = 0  # Burn-in period
    n_steps = int(n_samples)  # Number of steps

    S = np.diag(diagonal)
    num_J = 50
    ground_truth = [0, np.trace(np.linalg.inv(S)), 0]

    trace_jsj, trace_jsjs, trace_jssj = [], [], []
    MSE2 = np.zeros(num_J)

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
        trace_jsj.append(-np.trace(J@S@J))
        trace_jsjs.append(-np.trace(J@S@J@S))
        trace_jssj.append(-np.trace(J@S@S@J))
        # MSE2[i] = get_bias_var_av_sum(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, scaled_gaussian)
        MSE2[i] = get_MMD(S, J, x0, repeat, burn_in, n_steps, dt, ground_truth, scaled_gaussian, n_samples)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

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

    # Adjust figure layout to prevent overlapping
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot correlation JSJ and MSE
    axes[0].scatter(trace_jsj, MSE2, alpha=0.5)
    axes[0].set_xlabel('trace(-JSJ)', labelpad=10)
    axes[0].set_ylabel('MMD', labelpad=10)
    axes[0].set_title('Correlation between\n-tr(JSJ) and MSE', pad=10)

    # Plot correlation between JSJS and MSE 
    axes[1].scatter(trace_jsjs, MSE2, alpha=0.5)
    axes[1].set_xlabel('trace(-JSJS)', labelpad=10)
    axes[1].set_ylabel('MMD', labelpad=10)
    axes[1].set_title('Correlation between\n-tr(JSJS) and MSE', pad=10)

    # Plot correlation between JSSJ and MSE
    axes[2].scatter(trace_jssj, MSE2, alpha=0.5)
    axes[2].set_xlabel('trace(-JSSJ)', labelpad=10)
    axes[2].set_ylabel('MMD', labelpad=10)
    axes[2].set_title('Correlation between\n-tr(JSSJ) and MSE', pad=10)

    # Plot scatter of trace(-JSJ) vs trace(-JSJS)
    scatter = axes[3].scatter(trace_jsj, trace_jsjs, c=MSE2,
                            cmap='viridis',
                            alpha=0.5)
    cbar = plt.colorbar(scatter, ax=axes[3], label='MMD', pad=0.1)
    cbar.ax.set_ylabel('MMD', labelpad=10)
    axes[3].set_xlabel('trace(-JSJ)', labelpad=10)
    axes[3].set_ylabel('trace(-JSJS)', labelpad=10)
    axes[3].set_title('Correlation between\n-tr(JSJ) and -tr(JSJS)', pad=10)

    # Add spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Set a single title above all subplots with padding
    plt.suptitle(f'ESJD Correlation (point color ~ MMD)\ndiag={str(diagonal)[1:-1]}, dt={dt:.3f}, fixed_evalues={fixed_evalues}',
                y=1.2)

    
    plt.savefig('plottings/MMD/correlation_ESJD_MMD_{:d}-D Gaussian, diag={}, dt={:.3f}, fixed_evalues={}, repeat={}, n_samples={}.png'.format(dim, str(diagonal)[1:-1], dt, fixed_evalues, repeat, n_samples), format='png', dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    dimension = sys.argv[1]
    diagonal = sys.argv[2]
    dt = sys.argv[3]
    fixed_evalues = sys.argv[4]
    repeat = sys.argv[5]
    n_samples = sys.argv[6]
    main(dim=int(dimension), 
         diagonal=np.fromstring(diagonal, sep=','), 
         dt=float(dt),
         fixed_evalues=int(fixed_evalues),
         repeat=int(repeat),
         n_samples=int(n_samples))
