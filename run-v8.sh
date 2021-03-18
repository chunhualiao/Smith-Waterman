#!/bin/bash

make -f makefile-lassen-apollo clean 
make -f makefile-lassen-apollo omp_smithW-v8-apollo.out
# run 50 times to collect enough training data
# 10k to 1000k, step 20k
# or run up to 25 times only
# 40k to 1040k, step 40k
counter=""
for size in {40000..1040000..40000};
#for size in {10000..1000000..250000};
# it may never trigger model building, if repeating the same sizes which are already sampled!!
do
 let "counter += 1"
 echo "running count=$counter, problem size=$size"
  APOLLO_TRACE_CSV_FOLDER_SUFFIX="-sw-v8" APOLLO_CROSS_EXECUTION=1 ./omp_smithW-v8-apollo.out $size $size
done
