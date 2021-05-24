#!/bin/bash -xe
# -x will show the expanded commands
# -e abor on any error

EXE_FILE=omp_smithW-v1-refinedOrig.out
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NODE_NAME=`uname -n`
HARDWAREE_NAME=`uname -m`
rm $EXE_FILE
make ./$EXE_FILE

# data size ranges used in metadirective iwomp'19 paper
# fixed M: 2,000
#  varying N: 400 - 60,000

# run 50 times to collect enough training data
# 10k to 1000k, step 20k
# or run up to 25 times only
# 40k to 1040k, step 40k

#M_SIZE=2000, we use equal sizes
# using the actual core count on lasseb
export OMP_NUM_THREADS=48 # 48 for Corona, 44 for lassen
# for each N size
counter=""
#for n_size in {256..7000..256}; do
#for N_SIZE in {32..16000..512}; do  # roughly 25 data points
#for N_SIZE in {32..46000..512}; do  # roughly 100 data points
for N_SIZE in {32..26000..512}; do  # roughly 100 data points
 let "counter += 1"
# echo "running count=$counter, problem m_size=$M_SIZE n_size=$n_size"
 echo "running count=$counter, problem M=$N_SIZE N=$N_SIZE"

    for repeat in 1 2 3;   do
    echo " Repeat=$repeat"
       ./$EXE_FILE $N_SIZE $N_SIZE
    done

done
