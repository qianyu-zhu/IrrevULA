"""
single particle with irreversible perturbation, Gaussian case
compare the performance of different J under the spectral-optimal constraint
"""
from function import *

def main(dim, diagonal):
    # Sample multiple J matrices and compute their objective values
    n_samples = 1000
    opt_obj_vals_jsj = []
    opt_obj_vals_jsjs = []
    nopt_obj_vals_jsj = []
    nopt_obj_vals_jsjs = []
    
    S = np.diag(diagonal)
    
    # Sample and evaluate opt_J matrices
    for _ in range(n_samples):
        J = get_opt_J(S)
        obj_val_jsj = -np.trace(J @ S @ J)
        obj_val_jsjs = -np.trace(J @ S @ J @ S)
        opt_obj_vals_jsj.append(obj_val_jsj)
        opt_obj_vals_jsjs.append(obj_val_jsjs)
            
    # Sample and evaluate nopt_J matrices
    for _ in range(n_samples):
        J = get_nopt_J(S)
        obj_val_jsj = -np.trace(J @ S @ J)
        obj_val_jsjs = -np.trace(J @ S @ J @ S)
        nopt_obj_vals_jsj.append(obj_val_jsj)
        nopt_obj_vals_jsjs.append(obj_val_jsjs)
    
    # Create 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot distributions for optimal J
    axes[0,0].hist(opt_obj_vals_jsj, bins=50, alpha=0.7)
    axes[0,0].set_title('Distribution of -tr(JSJ) for optimal J')
    axes[0,0].set_xlabel('Objective Value')
    axes[0,0].set_ylabel('Count')
    
    axes[0,1].hist(opt_obj_vals_jsjs, bins=50, alpha=0.7)
    axes[0,1].set_title('Distribution of -tr(JSJS) for optimal J')
    axes[0,1].set_xlabel('Objective Value')
    axes[0,1].set_ylabel('Count')
    
    # Plot correlation for optimal J
    axes[0,2].scatter(opt_obj_vals_jsj, opt_obj_vals_jsjs, alpha=0.5)
    axes[0,2].set_title('Correlation for optimal J')
    axes[0,2].set_xlabel('-tr(JSJ)')
    axes[0,2].set_ylabel('-tr(JSJS)')
    
    # Plot distributions for non-optimal J
    axes[1,0].hist(nopt_obj_vals_jsj, bins=50, alpha=0.7)
    axes[1,0].set_title('Distribution of -tr(JSJ) for non-optimal J')
    axes[1,0].set_xlabel('Objective Value')
    axes[1,0].set_ylabel('Count')
    
    axes[1,1].hist(nopt_obj_vals_jsjs, bins=50, alpha=0.7)
    axes[1,1].set_title('Distribution of -tr(JSJS) for non-optimal J')
    axes[1,1].set_xlabel('Objective Value')
    axes[1,1].set_ylabel('Count')
    
    # Plot correlation for non-optimal J
    axes[1,2].scatter(nopt_obj_vals_jsj, nopt_obj_vals_jsjs, alpha=0.5)
    axes[1,2].set_title('Correlation for non-optimal J')
    axes[1,2].set_xlabel('-tr(JSJ)')
    axes[1,2].set_ylabel('-tr(JSJS)')
    
    plt.tight_layout()
    plt.savefig('plottings/objective_distributions_diag=[{}].png'.format(str(diagonal)[1:-1]), dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    dimension = sys.argv[1]
    diagonal = sys.argv[2]

    main(dim=int(dimension), 
         diagonal=np.fromstring(diagonal, sep=','))