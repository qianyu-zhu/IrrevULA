#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/manifold_blr

# echo "blr, fix length, non-adaptive"

# Define common parameters
# common_params="--M 128 --T 100 --path Jun_22_fixlength/Unperturbed"

# echo $common_params
# # Function to run simulations for a given method
# for dt in 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5; do
#     srun python -u main_blr_unperturbed.py $common_params --dt $dt
# done

srun python -u HM.py