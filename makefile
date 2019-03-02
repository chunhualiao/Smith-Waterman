# large size: 45 s for 1 thread on tux285
MSIZE=25600

# using C++ compiler to be more restrictive 	
#CC=g++-8
CC=g++
all: omp_smithW_debug.out omp_smithW_O3.out
clean:
	rm -rf *.out

omp_smithW_debug.out: parameters.h omp_smithW.c	
	$(CC) omp_smithW.c -o $@ -fopenmp -DDEBUG	

omp_smithW_O3.out: parameters.h omp_smithW.c	
	$(CC) -O3 omp_smithW.c -o $@ -fopenmp 

omp_smithW-v5-target.out: omp_smithW-v5-target.c parameters.h
	xlc-gpu  -qsmp -qoffload -o $@ $<

#verify the results
check: omp_smithW_O3.out
	./$< 

# this is the version I am interested in
check2: omp_smithW_O3.out
	./$< $(MSIZE) $(MSIZE)

