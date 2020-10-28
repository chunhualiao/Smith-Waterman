# Makefile for rotated Smith-Waterman

OUTPUTDIR:=$(shell hostname | sed -e s/[0-9]*//g)
NVCC:=nvcc
NVFLAGS:=-std=c++11 -Xcompiler -Wall -Xcompiler -Wextra

ifeq ($(MODE),debug)
  DBGFLAG:=-g -G
  NVFLAGS+=-O0
else
  DBGFLAG:=-DNDEBUG=1
  NVFLAGS+=-O3
endif

TARGETS := $(OUTPUTDIR)/smithW-cuda-rotated.bin 

.PHONY: all
all: $(TARGETS)

$(OUTPUTDIR)/smithW-cuda-rotated.bin: sw-rotated.cu
	$(NVCC) $(NVFLAGS) $(DBGFLAG) -DALLOC_ZERO=1 $< -o $@

.PHONY: clean
clean:
	rm -f $(OUTPUTDIR)/*bin
