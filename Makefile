#!/bin/make
RTE_DIR = rte
RRTMGP_DIR = rrtmgp

#
# Compiler variables FC, FCFLAGS can be set in the environment or in Makefile.conf
#
-include Makefile.conf
#
# Choose kernels depending on environment variable
#

ifeq ($(USE_OPENACC),1)
  RTE_KERNEL_DIR    = rte/kernels-openacc
  RRTMGP_KERNEL_DIR = rrtmgp/kernels-openacc
  FCFLAGS += -DUSE_OPENACC
endif

RTE_KERNEL_DIR += rte/kernels
RRTMGP_KERNEL_DIR += rrtmgp/kernels
NEURAL_DIR = neural
VPATH = $(RTE_DIR):$(RTE_KERNEL_DIR):$(RRTMGP_DIR):$(RRTMGP_KERNEL_DIR):$(NEURAL_DIR)

ifeq ($(DOUBLE_PRECISION),1)
	FCFLAGS += -DDOUBLE_PRECISION
endif

ifeq ($(FAST_EXPONENTIAL),1)
    # Approximation for exp function, significantly faster on non-intel compilers
	FCFLAGS += -DFAST_EXPONENTIAL
endif

# Use GPTL timing library? Added this to main makefile to enable
# timing instrumentation within RRTMGP and RTE source code
ifeq ($(GPTL_TIMING),1)
	# Timing library GPTL
	FCFLAGS += -I$(TIME_DIR)/include -DUSE_TIMING 

# Use GPTL with PAPI for hardware performance counters? (enables measuring compute intensity)
else ifeq ($(GPTL_TIMING),2)
	FCFLAGS += -I$(TIME_DIR)/include -DUSE_TIMING -DUSE_PAPI
endif

all: ../lib/librte.a ../lib/librrtmgp.a ../lib/libneural.a

COMPILE = $(FC) $(FCFLAGS) $(FCINCLUDE) -c
%.o: %.F90
	$(COMPILE) $<

include $(RTE_DIR)/Make.depends
include $(RRTMGP_DIR)/Make.depends
include $(NEURAL_DIR)/Make.depends

../lib/librte.a: $(RTE_SRC)
	ar r ../lib/librte.a $(RTE_SRC)

../lib/librrtmgp.a: $(RRTMGP_SRC)
	ar r ../lib/librrtmgp.a $(RRTMGP_SRC)

../lib/libneural.a: $(NEURAL_SRC)
	ar r ../lib/libneural.a $(NEURAL_SRC)

clean:
	rm -f *.optrpt *.mod *.o librrtmgp.a librte.a libneural.a