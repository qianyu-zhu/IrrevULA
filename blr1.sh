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

# echo "blr, fix length, non-adaptive"

# Define common parameters
# common_params="--M 128 --T 100 --path Aug_04_fixlength/Unperturbed"

# echo $common_params
# # Function to run simulations for a given method
# for dt in 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5; do
#     srun python -u main_blr_unperturbed.py $common_params --dt $dt
# done

echo "blr, fix budget, non-adaptive"

common_params="--M 256 --K 100000 --path Aug_04_fixbudget/Unperturbed"

N_stats=11
echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_unperturbed.py $common_params --dt $dt --N_stats $N_stats
done


common_params="--M 256 --K 100000 --path Aug_04_fixbudget/Irr-S"

echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_s.py $common_params --dt $dt --N_stats $N_stats
done

common_params="--M 256 --K 100000 --path Aug_04_fixbudget/Irr-M"

echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_m.py $common_params --dt $dt --N_stats $N_stats
done    

common_params="--M 256 --K 100000 --path Aug_04_fixbudget/Irr-L"

echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_l.py $common_params --dt $dt --N_stats $N_stats
done

common_params="--M 256 --K 100000 --path Aug_04_fixbudget/Irr-O"

echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_o.py $common_params --dt $dt --N_stats $N_stats
done


common_params="--M 256 --K 100000 --path Aug_04_fixbudget/Irr-SO"

echo $common_params
# Function to run simulations for a given method
for dt in 0.0001 0.00025 0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002; do
    srun python -u main_blr_irr_so.py $common_params --dt $dt --N_stats $N_stats    
done

