#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Loading the required module(s)
source /etc/profile
module load miniforge/24.3.0-0
source MCMC_env/bin/activate
cd /home/qianyu_z/MCMC/ICA

echo "ICA3, irr-o"

srun python -u main_bench_MH.py --n 3 --dt 0.00007 --T 50 --lambda_ 1 --num_chains 2 --subsample_rate 100 #--path results/MH --initial_point results/MH/main_giIrr_num_chains32_dt8e-05_T50.npy

# python -u main_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
# python -u main_opt_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
# python -u main_ir_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
# python -u main_nopt_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
