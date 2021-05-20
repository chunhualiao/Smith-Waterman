# large size: 45 s for 1 thread on tux285
MSIZE=25600

# using C++ compiler to be more restrictive 	
#CC=g++-8
# CC=clang # default compiler: still illegal memory access bug
CC=/g/g17/liao6/workspace-wsa/opt/llvm-master-offload/bin/clang++
FLAGS=-O3 -g -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -DDEBUG

all: omp_smithW-v6-target-inlined.out omp_smithW-v7-adaptive.out omp_smithW-v6.2-target-inlined.out
v1:omp_smithW-v1-refinedOrig.out
v7:omp_smithW-v7-adaptive.out
v5:omp_smithW-v5-target.out
v6:omp_smithW-v6-target-inlined.out
v6.2:omp_smithW-v6.2-target-inlined.out

#all: omp_smithW_debug.out omp_smithW_O3.out
clean:
	rm -rf *.out *.core
# my own build of compiler
LINK_FALGS=-Wl,--rpath,/g/g17/liao6/workspace-wsa/opt/llvm-master-offload/lib	
omp_smithW_debug.out: parameters.h omp_smithW.c	
	$(CC) omp_smithW.c $(LINK_FALGS) -o $@ -fopenmp -DDEBUG	

omp_smithW-v1-refinedOrig.out: omp_smithW-v1-refinedOrig.c parameters.h
	$(CC) $(FLAGS) $(LINK_FALGS) -o $@ $<
omp_smithW_O3.out: parameters.h omp_smithW.c	
	$(CC) -O3 omp_smithW.c $(LINK_FALGS) -o $@ -fopenmp 

omp_smithW-v7-adaptive.out: omp_smithW-v7-adaptive.c parameters.h
	$(CC) $(FLAGS) $(LINK_FALGS) -g  -o $@ $<
hasGPU.out: hasGPU.c
	$(CC) $(FLAGS) -o $@ $<
omp_smithW-v5-target.out: omp_smithW-v5-target.c parameters.h
	$(CC) $(FLAGS) $(LINK_FALGS) -o $@ $<
omp_smithW-v6-target-inlined.out: omp_smithW-v6-target-inlined.c parameters.h
	$(CC) $(FLAGS) $(LINK_FALGS) -o $@ $<
omp_smithW-v6.2-target-inlined.out: omp_smithW-v6.2-target-inlined.c parameters.h	
	$(CC) $(FLAGS) $(LINK_FALGS) -o $@ $<
# not working: parsing error for clang-gpu	
#	clang-gpu -g -o $@ $<
#	xlc-gpu -DDEBUG -g -qsmp -qoffload -o $@ $<

#verify the results
check: omp_smithW-v7-adaptive.out
	./$< $(MSIZE) $(MSIZE) 

# this is the version I am interested in
check2: omp_smithW_O3.out
	./$< $(MSIZE) $(MSIZE)

