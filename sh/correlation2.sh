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
python -u correlation_ESJD_MSE_separate.py 3 "1,$second_entry,$third_entry" 0.1 0
#     done
# done

 