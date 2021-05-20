
# APOLLO_HOME is your environment variable indicating the path to Apollo Installation
# e.g.: 
#   export APOLLO_HOME=/g/g17/liao6/workspace-wsa/parco-ldrd/apollo/install-lassen
APOLLO_DIR=$(APOLLO_HOME)
INC_DIR=$(APOLLO_DIR)/include 
LIB_DIR=$(APOLLO_DIR)/lib
LIBS=-lapollo

# large size: 45 s for 1 thread on tux285
MSIZE=25600

# using C++ compiler to be more restrictive 	
#CC=g++-8
CC=clang++
FLAGS=-O3 -g -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
all: omp_smithW-v8-apollo.out
clean:
	rm -rf *.out *.core

omp_smithW-v8-apollo.out: omp_smithW-v8-apollo.cpp parameters.h
	$(CC) $(FLAGS) -I $(INC_DIR) -L $(LIB_DIR) -Wl,--rpath,$(LIB_DIR) $<  -o $@ $(LIBS)

#verify the results
check: omp_smithW-v8-apollo.out
	APOLLO_TRACE_CSV_FOLDER_SUFFIX="-sw-v8" APOLLO_CROSS_EXECUTION=1 ./$< $(MSIZE) $(MSIZE) 

