#!/bin/bash

# Job Flags
#SBATCH --job-name=ica1
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

echo "ICA1, unperturbed"

srun python -u main_bench_MH.py --n 2 --dt 0.00008 --T 200 --lambda_ 1 --num_chains 1 --subsample_rate 100 --path results/MH #--initial_point results/MH/main_giIrr_num_chains32_dt5e-05_T50.npy

# srun python -u main_unperturbed.py --n 1 --dt 0.0001 --T 1000 --lambda_ 1 --num_chains 32 --subsample_rate 250 --path results/unperturbed

