#include "strata/host/ds41_prefill_wire.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void) {
    strata_ds41_prefill_request r;
    strata_ds41_prefill_state state;
    strata_ds41_prefill_capabilities peer;
    unsigned i;
    memset(&r, 0, sizeof r);
    memset(&state, 0, sizeof state);
    memset(&peer, 0, sizeof peer);
    peer.magic = STRATA_DS41_PREFILL_MAGIC;
    peer.version = STRATA_DS41_PREFILL_VERSION;
    peer.flags = STRATA_DS41_PREFILL_REQUIRED_CAPS;
    peer.max_tokens = 8192; peer.max_routes = 131072;
    peer.max_hidden = 5120; peer.max_top_k = 16;
    CHECK(strata_ds41_prefill_negotiate(&peer, 8192, 5120, 6));
    for (i = 0; i < 3; ++i) {
        peer.flags ^= 1u << i;
        CHECK(!strata_ds41_prefill_negotiate(&peer, 8192, 5120, 6));
        peer.flags ^= 1u << i;
    }
    peer.version++;
    CHECK(!strata_ds41_prefill_negotiate(&peer, 8192, 5120, 6));
    peer.version--;
    CHECK(!strata_ds41_prefill_negotiate(&peer, UINT32_MAX, 5120, 6));
    CHECK(!strata_ds41_prefill_negotiate(NULL, 8192, 5120, 6));

    r.version = 1; r.op = STRATA_DS41_PREFILL_EXECUTE; r.sequence = 7;
    r.tokens = 8192; r.routes = 49152; r.hidden = 5120; r.top_k = 6;
    r.activation = STRATA_DS41_PREFILL_Q8_1;
    r.activation_bytes = (uint64_t)8192 * 160 * 36;
    CHECK(strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));
    CHECK(!strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 8));
    r.first_token = 8193;
    CHECK(!strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));
    r.first_token = 0; r.activation_bytes--;
    CHECK(!strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));
    r.activation_bytes++; r.reserved = 1;
    CHECK(!strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));
    r.reserved = 0; r.routes = 0;
    CHECK(strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));
    r.routes = 49153;
    CHECK(!strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));
    r.tokens = UINT32_MAX;
    CHECK(!strata_ds41_prefill_validate_request(&r, 40, 5120, 16384, 7));

    /* Repeated lending/restoration, replay rejection, and decode interlock. */
    for (i = 0; i < 100; ++i) {
        memset(&r, 0, sizeof r);
        r.version = 1; r.hidden = 5120; r.sequence = state.sequence;
        r.op = STRATA_DS41_PREFILL_END;
        CHECK(!strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        r.op = STRATA_DS41_PREFILL_BEGIN;
        CHECK(strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        CHECK(!strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        r.sequence = state.sequence;
        CHECK(!strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        r.op = STRATA_DS41_PREFILL_DECODE_RAW;
        r.tokens = 1; r.top_k = 6; r.routes = 6;
        r.activation = STRATA_DS41_PREFILL_F32; r.activation_bytes = 5120 * 4;
        CHECK(!strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        r.op = STRATA_DS41_PREFILL_EXECUTE;
        r.activation = STRATA_DS41_PREFILL_Q8_1; r.activation_bytes = 160 * 36;
        CHECK(strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        memset(&r, 0, sizeof r);
        r.version = 1; r.hidden = 5120; r.sequence = state.sequence;
        r.op = STRATA_DS41_PREFILL_END;
        CHECK(strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
        r.sequence = state.sequence; r.op = STRATA_DS41_PREFILL_DECODE_RAW;
        r.tokens = 64; r.top_k = 6; r.routes = 384;
        r.activation = STRATA_DS41_PREFILL_F32; r.activation_bytes = 64 * 5120 * 4;
        CHECK(strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
    }
    state.sequence = UINT64_MAX; r.sequence = UINT64_MAX;
    CHECK(!strata_ds41_prefill_advance(&state, &r, 40, 5120, 16384));
    CHECK(state.sequence == UINT64_MAX && !state.active);
    {
        unsigned char native[7 * 4 * 144 + 128];
        unsigned char wire[7 * 3 * 144];
        unsigned char restored[sizeof native];
        size_t j;
        for (j = 0; j < sizeof native; ++j) native[j] = (unsigned char)(j * 137 + j / 144);
        memset(wire, 0xa5, sizeof wire);
        CHECK(!strata_ds41_prefill_q8_repack(wire, sizeof wire - 1,
            native, sizeof native, 7, 384, 512, 1));
        CHECK(wire[0] == 0xa5);
        CHECK(strata_ds41_prefill_q8_repack(wire, sizeof wire,
            native, sizeof native, 7, 384, 512, 1));
        CHECK(strata_ds41_prefill_q8_repack(restored, sizeof restored,
            wire, sizeof wire, 7, 384, 512, 0));
        CHECK(!memcmp(native, restored, 7 * 3 * 144));
        for (j = 7 * 3 * 144; j < sizeof restored; ++j) CHECK(!restored[j]);
        CHECK(!memcmp(wire + 3 * 144, native + 144, 144));
        CHECK(!memcmp(wire + 144, native + 7 * 144, 144));
        CHECK(!strata_ds41_prefill_q8_repack(wire, sizeof wire,
            native, sizeof native, 7, 384, 384, 1));
    }
    puts("prefill wire validation passed");
    return 0;
}
