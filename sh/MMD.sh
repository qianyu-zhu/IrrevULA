#!/bin/bash

# Loading the required module(s)

source /etc/profile
module load anaconda/2023a
source activate MCMC


# Loop through different diagonal arrays where third entry varies from 0.1 to 0.5
# for i in {1..5}; do
#     for j in $(seq $i 5); do
second_entry=$(echo "0.5" | bc -l)
third_entry=$(echo "0.2" | bc -l)
echo "Running with diagonal array: [1,$second_entry,$third_entry]"

# n_samples=2000
i=5
# for i in 1 5 10 20; do
for n_samples in 1000 2000 3000 4000 5000; do
    echo "Running with repeat=$i, n_samples=$n_samples"
    
    python -u correlation_ESJD_MSE.py 3 "1,$second_entry,$third_entry" 0.1 1 $i $n_samples # [dimension, diagonal, dt, fixed_evalues, repeat, n_samples]
    python -u correlation_ESJD_MSE.py 3 "1,$second_entry,$third_entry" 0.1 0 $i $n_samples
done
#     done
# done

 