#!/bin/bash

# Job Flags
#SBATCH --job-name=ica2
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

echo "ICA2, unperturbed"

srun python -u main_bench_MH.py --n 3 --dt 0.0001 --T 200 --lambda_ 1 --num_chains 1 --subsample_rate 100 #--path results/MH --initial_point results/MH/main_giIrr_num_chains32_dt2e-05_T50.npy

# srun python -u main_irr-so.py --n 1 --dt 0.0001 --T 1000 --lambda_ 1 --num_chains 32 --subsample_rate 250 --path results/irr-so
# python -u main_giIrr.py --n 2 --dt 0.0001 --T 10 --lambda_ 1 --num_chains 1 --subsample_rate 500 --path statistics/test
