# large size: 45 s for 1 thread on tux285
MSIZE=25600

# AMD GPU
ifeq ($(findstring corona,$(HOSTNAME)), corona)
  CC=/opt/rocm-4.1.0/llvm/bin/clang++
#  CXXFLAGS=-O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-version=50 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

  BASE_FLAGS=-O3 -g -std=c++11 -DSKIP_BACKTRACK=1 #-DDEBUG
  OFFLOADING_FLAGS=-target x86_64-pc-linux-gnu -fopenmp -fopenmp-version=50 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906
# on lassen, Nvidia GPU  
else
  # using C++ compiler to be more restrictive 	
  #CC=g++-8
  CC=clang++ # default clang 11 compiler: still illegal memory access bug # crashes on 6.1
#  CC=/g/g17/liao6/workspace-wsa/opt/llvm-master-offload/bin/clang++ # clang 12.0.x
  BASE_FLAGS=-O3 -g -std=c++11 -DSKIP_BACKTRACK=1
  OFFLOADING_FLAGS=-fopenmp-targets=nvptx64-nvidia-cuda
# my own build of compiler
#LINK_FALGS=-L/g/g17/liao6/workspace-wsa/parco-ldrd/apollo/install-lassen/lib -Wl,--rpath,/g/g17/liao6/workspace-wsa/opt/llvm-master-offload/lib	
#  LINK_FALGS=-Wl,--rpath,/g/g17/liao6/workspace-wsa/opt/llvm-master-offload/lib	
endif

# OpenMP offloading versions	
SRC_FILES = \
  hasGPU.cpp \
  omp_smithW-v5-target.cpp \
  omp_smithW-v6.1-target-inlined.cpp \
  omp_smithW-v6.2-target-inlined.cpp \
  omp_smithW-v6.3-target-inlined.cpp \
  omp_smithW-v7-adaptive.cpp


all: omp_smithW-v6-target-inlined.out omp_smithW-v7-adaptive.out omp_smithW-v6.2-target-inlined.out
v1:omp_smithW-v1-refinedOrig.out
v7:omp_smithW-v7-adaptive.out
v5:omp_smithW-v5-target.out
v6:omp_smithW-v6-target-inlined.out
v6.2:omp_smithW-v6.2-target-inlined.out

all: v0-serial_smithW.out omp_smithW-v1-refinedOrig.out omp_smithW-v6-target-inlined.out omp_smithW-v6.2-target-inlined.out
clean:
	rm -rf *.out *.core

# serial version	
# compiling without -fopenmp	
v0-serial_smithW.out: omp_smithW-v1-refinedOrig.cpp
	$(CC) $(BASE_FLAGS) -o $@ $<

# OMP CPU threading versions

omp_smithW-v1-refinedOrig.out: omp_smithW-v1-refinedOrig.cpp parameters.h
	$(CC) $(BASE_FLAGS) -fopenmp $(LINK_FALGS) -o $@ $<

omp_smithW_O3.out: parameters.h omp_smithW.c	
	$(CC) -O3 omp_smithW.c $(LINK_FALGS) -o $@ -fopenmp 

%.out: %.cpp
	$(CC) $(BASE_FLAGS) -fopenmp $(OFFLOADING_FLAGS) $(LINK_FALGS) -o $@ $<
# not working: parsing error for clang-gpu	
#	clang-gpu -g -o $@ $<
#	xlc-gpu -DDEBUG -g -qsmp -qoffload -o $@ $<

#verify the results
check: omp_smithW-v7-adaptive.out
	./$< $(MSIZE) $(MSIZE) 

# this is the version I am interested in
check2: omp_smithW_O3.out
	./$< $(MSIZE) $(MSIZE)

