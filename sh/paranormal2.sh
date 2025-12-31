#!/bin/bash
#SBATCH --exclusive


# Loading the required module(s)
source /etc/profile
module load anaconda/2023a
source activate MCMC
cd /home/gridsan/qzhu/MCMC/manifold_params_normal_2

echo "paranormal2"

# python -u main.py --M 1 --h 0.005 --T 500
python -u main.py --M 50 --h 0.001 --T 10000 --path mar_18
# python -u main.py --M 100 --h 0.04 --T 10000
# python -u main.py --M 50 --h 0.01 --T 2000
# python -u main.py --M 100 --h 0.0075 --T 100

# python -u main.py --M 100 --h 0.005 --T 400
# python -u main.py --M 100 --h 0.01 --T 400