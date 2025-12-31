#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/ellipse

echo "ellipse, fix length, adaptive"

# Define common parameters
common_params="--M 512 --T 4000 --path Sep_09_adapt_fixlength"

echo $common_params
# Function to run simulations for a given method
for h in 0.04 0.08 0.12 0.16 0.2 0.24 0.28 0.32 0.36 0.4; do
    echo "h = $h"
    srun python -u main_adapt_fixlength.py $common_params --h $h
done

# echo "ellipse, fix budget, non-adaptive"

# # Define common parameters
# common_params="--M 1024 --K 100000 --path Jul_01_fixbudget"

# echo $common_params
# # Function to run simulations for a given method
# for h in 0.02 0.04 0.08 0.12 0.16 0.2 0.24 0.28 0.32 0.36 0.4; do
#     echo "h = $h"
#     srun python -u main_fixbudget.py $common_params --h $h
# done


# Run simulations for RM method (only for h=0.01)
# srun python -u main_adapt_fixbudget.py $common_params --h 0.02 --method Standard

# Run simulations for nopt and opt methods
# run_simulations Standard
# run_simulations irr
# run_simulations RM
# run_simulations nopt
# run_simulations opt