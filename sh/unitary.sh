#!/bin/bash
#SBATCH --exclusive


# Loading the required module(s)
source /etc/profile
module load anaconda/2023a
source activate MCMC
cd /home/gridsan/qzhu/MCMC/unitary

echo "unitary"

python -u main.py --M 100 --h 0.01 --T 10000 --sigma_true 1 1 1 1
python -u main.py --M 100 --h 0.03 --T 10000 --sigma_true 1 1 1 1
# python -u main.py --M 100 --h 0.04 --T 10000
# python -u main.py --M 100 --h 0.05 --T 10000