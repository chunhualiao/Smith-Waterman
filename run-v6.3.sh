#!/bin/bash -xe
# -x will show the expanded commands
# -e abor on any error

EXE_FILE=omp_smithW-v6.3-target-inlined.out
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
#export OMP_NUM_THREADS=44 # corona count, # lassen should be 44 threads
# for each N size
counter=""
#for n_size in {256..7000..256}; do
#for N_SIZE in {32..16000..512}; do  # roughly 25 data points
#for N_SIZE in {32..46000..512}; do  # roughly 100 data points
#for N_SIZE in {32..26000..512}; do  # roughly 100 data points
for N_SIZE in {2..150000..5000}; do  # larger size
 let "counter += 1"
# echo "running count=$counter, problem m_size=$M_SIZE n_size=$n_size"
    for repeat in 1 2 3 ;   do
    echo " Repeat=$repeat"
       ./$EXE_FILE 2000 $N_SIZE
    done

done
