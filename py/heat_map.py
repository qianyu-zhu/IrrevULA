"""
plot the heat-map for MSE^2 given fixed budget and varying step-size/magnitude of J in 2-D
"""
from function import *

def get_mse2(S, J, repeat, budget, dt, ground_truth, gradient):
    dim = len(S)
    
    for j in tqdm(range(repeat)):
        x0 = np.random.multivariate_normal(np.zeros(dim), np.linalg.inv(S))
        trajectory = langevin_dynamics_high_dim(x0, gradient, dt, budget, S, J)
        func_1 = np.sum(trajectory, axis = 1) # x+y
        func_2 = np.linalg.norm(trajectory, axis = 1)**2 # x**2 + y**2 - 11
        func_3 = np.prod(trajectory, axis = 1) # x*y
        estimator = np.array([np.mean(func_1), np.mean(func_2), np.mean(func_3)])
        mse2 = (estimator - ground_truth)**2
    return mse2 # returna vector of mse2
        


def main(budget, diagonal, repeat, combined):
    dim = len(diagonal)
    S = np.diag(diagonal)
    ground_truth = np.array([0, np.trace(np.linalg.inv(S)), 0])
    J_list = [1, 2, 3, 4, 5, 6, 7, 8]
    dt_list = [1/100, 1/50, 1/20, 1/10, 1/5]
    map = np.zeros((len(J_list), len(dt_list), 3))
    for i, irr in enumerate(J_list):
        J = np.array([[0,irr],[-irr,0]])
        for j, dt in enumerate(dt_list):
            map[i, j, :] = get_mse2(S, J, repeat, budget, dt, ground_truth, scaled_gaussian)

    # Add x and y axis labels with actual values
    if not combined:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax in axes:
            ax.set_xticks(np.arange(len(dt_list)))
            ax.set_yticks(np.arange(len(J_list)))
            ax.set_xticklabels([f'{dt:.3f}' for dt in dt_list])
            ax.set_yticklabels([f'{j}' for j in J_list])
            for i in range(3):
                im = axes[i].imshow(np.sqrt(map[:,:,i]), cmap='hot', interpolation='nearest')
                axes[i].set_title(f'MSE^2 for function {i+1}')
                axes[i].set_xlabel('dt index')
                axes[i].set_ylabel('J magnitude index') 
        for i in range(3):
            plt.colorbar(im, ax=axes[i])
        plt.suptitle('budget = ' + str(budget) + ', diag = ' + str(diagonal) + ', repeat = ' + str(repeat))
        plt.savefig('plottings/heat_map_budget={}_diag={}_repeat={}.png'.format(budget, diagonal, repeat), format='png', dpi=200)
        plt.close()
    else:
        # Combine all three statistics into one plot
        plt.figure(figsize=(8, 6))
        combined_map = np.mean(map, axis=2)  # Average across the 3 statistics
        
        plt.xticks(np.arange(len(dt_list)), [f'{dt:.3f}' for dt in dt_list])
        plt.yticks(np.arange(len(J_list)), [f'{j}' for j in J_list])
        
        im = plt.imshow(combined_map, cmap='hot', interpolation='nearest')
        plt.title('Combined MSE^2 across all functions, budget = ' + str(budget) + ', diag = ' + str(diagonal) + ', repeat = ' + str(repeat))
        plt.xlabel('dt')
        plt.ylabel('J magnitude')
        plt.colorbar(im)
        plt.savefig('plottings/combined_heat_map_budget={}_diag={}_repeat={}.png'.format(budget, diagonal, repeat), format='png', dpi=200)
        plt.close()
    return

if __name__ == '__main__':
    budget = int(sys.argv[1])
    diagonal = [1, 1/20]
    repeat = int(sys.argv[2])
    combined = int(sys.argv[3]) # 0 for not combined, 1 for combined
    main(budget, diagonal, repeat, combined)


    # python heat_map.py 5000 1 0 '1,0.05'