#!/bin/bash
#SBATCH --exclusive


# Loading the required module(s)
source /etc/profile
module load anaconda/2023a
source activate MCMC
cd /home/gridsan/qzhu/MCMC/banana

echo "banana2"

python main_complex.py --M 200 --h 0.01 --T 4000 --b 0.0 --path mar_18
# python main.py --M 200 --h 0.01 --T 10000 --b 0.0 --path mar_17_3
# python main.py --M 200 --h 0.015 --T 10000 --b 0.0 --path mar_17_3
# python main.py --M 200 --h 0.02 --T 10000 --b 0.0 --path mar_17_3
# python main.py --M 200 --h 0.025 --T 10000 --b 0.0 --path mar_17_3

python main_complex.py --M 200 --h 0.01 --T 4000 --b 0.1 --path mar_18
# python main.py --M 200 --h 0.01 --T 10000 --b 0.15 --path mar_17_3
# python main.py --M 200 --h 0.015 --T 10000 --b 0.15 --path mar_17_3
# python main.py --M 200 --h 0.02 --T 10000 --b 0.15 --path mar_17_3
# python main.py --M 200 --h 0.025 --T 10000 --b 0.15 --path mar_17_3

python main_complex.py --M 200 --h 0.01 --T 4000 --b 0.2 --path mar_18
# python main.py --M 200 --h 0.01 --T 10000 --b 0.3 --path mar_17_3
# python main.py --M 200 --h 0.015 --T 10000 --b 0.3 --path mar_17_3
# python main.py --M 200 --h 0.02 --T 10000 --b 0.3 --path mar_17_3
# python main.py --M 200 --h 0.025 --T 10000 --b 0.3 --path mar_17_3

python main_complex.py --M 200 --h 0.01 --T 4000 --b 0.3 --path mar_18
# python main.py --M 200 --h 0.01 --T 10000 --b 0.6 --path mar_17_3
# python main.py --M 200 --h 0.015 --T 10000 --b 0.6 --path mar_17_3
# python main.py --M 200 --h 0.02 --T 10000 --b 0.6 --path mar_17_3
# python main.py --M 200 --h 0.025 --T 10000 --b 0.6 --path mar_17_3


