#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/multi_gaussian

K=200000
common_params="--M 128 --K $K --path June_16_K_$K"
echo "multi_gaussian 2, fix budget, nonadaptive"
echo $common_params

# Function to run simulations for a given method
for h in 0.2 0.35 0.4; do
    srun python -u main_fixbudget.py $common_params --h $h
done

echo "done"

