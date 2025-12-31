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
common_params="--M 256 --T 100 --path Aug_04_fixlength/Unperturbed"
N_stats=11
echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_unperturbed.py $common_params --dt $dt --N_stats $N_stats
done

# Define common parameters
common_params="--M 256 --T 100 --path Aug_04_fixlength/Irr-S"
N_stats=11
echo $common_params
echo "Irr-S"
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_s.py $common_params --dt $dt --N_stats $N_stats
done


N_stats=11
common_params="--M 256 --T 100 --path Aug_04_fixlength/Irr-M"
echo "Irr-M"
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_m.py $common_params --dt $dt --N_stats $N_stats
done

common_params="--M 256 --T 100 --path Aug_04_fixlength/Irr-L"
echo "Irr-L"
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_l.py $common_params --dt $dt --N_stats $N_stats
done

# Define common parameters
common_params="--M 256 --T 100 --path Aug_04_fixlength/Irr-O"

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
