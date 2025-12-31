#!/bin/bash

# Job Flags
#SBATCH --job-name=ica
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/ICA

echo "ICA, unperturbed"

srun python -u main_bench_MH.py --n 1 --dt 0.00009 --T 200 --lambda_ 1 --num_chains 1 --subsample_rate 100 --path results/MH #--initial_point results/MH/main_giIrr_num_chains32_dt1e-04_T50.npy

# To start from stored chains, use --initial_point argument:
# srun python -u main_bench_MH.py --n 2 --dt 0.00001 --T 250 --lambda_ 1 --num_chains 32 --subsample_rate 100 --path results/MH --initial_point results/MH/main_giIrr_num_chains32_dt0.00001_T250_lambda_1_1.npy

# To start from random initial points, do not use --initial_point argument:
# srun python -u main_unperturbed.py --n 1 --dt 0.0001 --T 1000 --lambda_ 1 --num_chains 32 --subsample_rate 250 --path results/unperturbed

