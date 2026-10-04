# ucomm: plain C11, no verbs headers or libraries needed (libibverbs is dlopen'ed at run time).
CC ?= cc
CFLAGS ?= -O2 -g
UCFLAGS = -std=c11 -Wall -Wextra -pedantic -Iinclude -Isrc -fPIC
LDLIBS = -ldl -lpthread
B = build

SRC = src/ucomm.c src/ucomm_boot.c src/ucomm_udp.c src/ucomm_ib.c src/ibvew.c
OBJ = $(SRC:src/%.c=$(B)/%.o)

all: $(B)/libucomm.a $(B)/libucomm.so $(B)/test_loopback $(B)/ucomm_bench

$(B)/%.o: src/%.c src/ucomm_internal.h src/ibvew.h include/ucomm.h | $(B)
	$(CC) $(UCFLAGS) $(CFLAGS) -c $< -o $@
$(B)/libucomm.a: $(OBJ)
	$(AR) rcs $@ $^
$(B)/libucomm.so: $(OBJ)
	$(CC) -shared -o $@ $^ $(LDLIBS)
$(B)/test_loopback: tests/test_loopback.c $(B)/libucomm.a
	$(CC) $(UCFLAGS) $(CFLAGS) $< $(B)/libucomm.a $(LDLIBS) -o $@
$(B)/ucomm_bench: tools/ucomm_bench.c $(B)/libucomm.a
	$(CC) $(UCFLAGS) $(CFLAGS) $< $(B)/libucomm.a $(LDLIBS) -o $@
$(B):
	mkdir -p $(B)

test: $(B)/test_loopback
	$(B)/test_loopback 2 && $(B)/test_loopback 3 && $(B)/test_loopback 4
	UCOMM_UDP_DROP=0.05 $(B)/test_loopback 3

clean:
	rm -rf $(B)
.PHONY: all test clean
