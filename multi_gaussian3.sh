#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/multi_gaussian

T=10000
common_params="--M 128 --T $T --path Jun_16_T_$T"
echo "multi_gaussian 3, fix length, nonadaptive"
echo $common_params

# Function to run simulations for a given method
for h in 0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
    srun python -u main_fixlength.py $common_params --h $h
done
