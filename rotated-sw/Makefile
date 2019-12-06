


OUTPUTDIR:=$(shell hostname | sed -e s/[0-9]*//g)
NVCC:=nvcc
NVFLAGS:=-std=c++11 -O3

ifeq ($(MODE),debug)
  DBGFLAG:=-g -G
else
  DBGFLAG:=-DNDEBUG=1
endif

TARGETS := $(OUTPUTDIR)/smithW-cuda.bin \
           $(OUTPUTDIR)/smithW-cuda-0.bin

all: $(TARGETS)

$(OUTPUTDIR)/smithW-cuda.bin: sw-rotated.cu
	$(NVCC) $(NVFLAGS) $(DBGFLAG) $< -o $@

$(OUTPUTDIR)/smithW-cuda-0.bin: sw-rotated.cu
	$(NVCC) $(NVFLAGS) $(DBGFLAG) -DALLOC_ZERO=1 $< -o $@

