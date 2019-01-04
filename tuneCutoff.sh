#!/bin/bash
# run experiments

for para1 in 1 2 4 8 16 32 64 128 256 512 1024
do
echo "--------- configuration: factor=$para1--------"
  sed -i "s/^#define CUTOFF .*/#define CUTOFF $para1/" parameters.h
  make omp_smithW_O3.out
./omp_smithW_O3.out 25600 25600
./omp_smithW_O3.out 25600 25600
./omp_smithW_O3.out 25600 25600

done

