#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/manifold_blr

echo "blr, fix length, non-adaptive"

# Define common parameters
common_params="--M 256 --T 100 --path Aug_04_fixlength/Irr-O"

N_stats=11
echo $common_params
echo "Irr-O"
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_o.py $common_params --dt $dt --N_stats $N_stats
done

common_params="--M 256 --T 100 --path Aug_04_fixlength/Irr-SO"
echo "Irr-SO"
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_so.py $common_params --dt $dt --N_stats $N_stats
done
