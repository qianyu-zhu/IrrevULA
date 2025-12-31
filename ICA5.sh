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

echo "ICA5, irr-m"

srun python -u main_irr-m.py --n 1 --dt 0.0001 --T 1000 --lambda_ 1 --num_chains 32 --subsample_rate 250 --path results/irr-m

# python -u main_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
# python -u main_opt_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
# python -u main_ir_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
# python -u main_nopt_short.py --n 1 --dt 0.00004 --T 1000 --lambda_ 1 --num_chains 25 --subsample_rate 250 --path statistics/short
