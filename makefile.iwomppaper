# large size: 45 s for 1 thread on tux285
MSIZE=25600

# using C++ compiler to be more restrictive 	
clean:
	rm -rf *.out *.core

C_SOURCE_FILES = \
	hasGPU.c \
	omp_smithW-v1-refinedOrig.c \
	omp_smithW-v2-ifClause.c \
	omp_smithW-v3-master-ompfor.c \
	omp_smithW-v4-parallel-serial.c \
	omp_smithW-v5-target.c \
	omp_smithW-v6-target-inlined.c \
	omp_smithW-v6.2-target-inlined.c \
	omp_smithW-v7-adaptive.c

%.out: %.c
	clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o $@ $<
#	xlc-gpu -o $@ $<
#	gcc -fopenmp -foffload="-lm" -lm -o $@ $<
#	clang-gpu -g -o $@ $<
#	xlc -g -qsmp -qoffload -o $@ $<

CC=g++
omp_smithW_debug.out: parameters.h omp_smithW.c	
	$(CC) omp_smithW.c -o $@ -fopenmp -DDEBUG	

omp_smithW_O3.out: parameters.h omp_smithW.c	
	$(CC) -O3 omp_smithW.c -o $@ -fopenmp 

# not working: parsing error for clang-gpu	
#	gcc -fopenmp  -DDEBUG
#	xlc-gpu -g -qsmp -qoffload -o $@ $<
#	clang-gpu -g -o $@ $<
#	xlc-gpu -DDEBUG -g -qsmp -qoffload -o $@ $<

#verify the results
check: omp_smithW_O3.out
	./$< 

checkgpu: hasGPU.out
	./$< 
# this is the version I am interested in
checkv1: omp_smithW-v1-refinedOrig.out
	./$< #$(MSIZE) $(MSIZE)

checkv2: omp_smithW-v2-ifClause.out
	./$< #$(MSIZE) $(MSIZE)

checkv3: omp_smithW-v3-master-ompfor.out
	./$< #$(MSIZE) $(MSIZE)
checkv4: omp_smithW-v4-parallel-serial.out
	./$< #$(MSIZE) $(MSIZE)
check2: omp_smithW_O3.out
	./$< $(MSIZE) $(MSIZE)
 
checkv5: omp_smithW-v5-target.out
	./$< # $(MSIZE) $(MSIZE)

checkv6: omp_smithW-v6-target-inlined.out
	./$< #$(MSIZE) $(MSIZE)

checkv6.2: omp_smithW-v6.2-target-inlined.out
	./$< # $(MSIZE) $(MSIZE)

checkv7: omp_smithW-v7-adaptive.out
	./$< $(MSIZE) $(MSIZE)
