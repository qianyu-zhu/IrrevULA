#!/bin/bash
#SBATCH --exclusive


# Loading the required module(s)
source /etc/profile
module load anaconda/2023a
source activate MCMC
cd /home/gridsan/qzhu/MCMC/ellipse
# cd /home/gridsan/qzhu/MCMC/manifold_params_normal_2

echo "paranormal"

# python -u main.py --M 100 --h 0.01 --T 10000
python -u main.py --M 100 --h 0.04 --T 50000 --path mar_16
# python -u main.py --M 100 --h 0.04 --T 10000
# python -u main.py --M 100 --h 0.05 --T 10000