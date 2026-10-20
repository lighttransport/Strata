# strata-pfetch: C11 + MPI (MPICC: mpicc here, mpifcc -Nclang natively on A64FX, mpifccpx -Nclang cross);
# strata-pput: the same engine without MPI (one rank, plus the --write direction).
MPICC ?= mpicc
CC ?= cc
CFLAGS ?= -O2 -g
PFLAGS = -std=c11 -Wall -Wextra -pedantic
LDLIBS = -lpthread
B = build
SRC = strata_pfetch.c pf_plan.c pf_hash.c
HDR = pf_plan.h pf_hash.h

all: $(B)/strata-pfetch $(B)/strata-pput

$(B)/strata-pfetch: $(SRC) $(HDR) | $(B)
	$(MPICC) $(PFLAGS) $(CFLAGS) $(SRC) $(LDLIBS) -o $@
$(B)/strata-pput: $(SRC) $(HDR) | $(B)
	$(CC) $(PFLAGS) $(CFLAGS) -DPFETCH_NO_MPI $(SRC) $(LDLIBS) -o $@
$(B):
	mkdir -p $(B)

test: all
	python3 test_pfetch.py

clean:
	rm -rf $(B)
.PHONY: all test clean
