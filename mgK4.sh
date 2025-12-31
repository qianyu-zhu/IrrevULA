#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/multi_gaussian

# T=1000
# common_params="--M 128 --T $T --path May_31_T_$T"
# echo "multi_gaussian, fix length, nonadaptive"
# echo $common_params

# # Function to run simulations for a given method
# for h in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#     srun python -u main_fixlength.py $common_params --h $h
# done

# echo "done"

K=1000000
for h in 0.3 0.35; do
    srun python -u fixbudget_mixing.py --K $K --h $h --path Jun_29_K_$K/traj
done

echo "done"