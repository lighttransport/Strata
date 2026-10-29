#include "strata/host/ds41_prefill_wire.h"
#include <string.h>

int strata_ds41_prefill_negotiate(
    const strata_ds41_prefill_capabilities *peer,
    uint32_t tokens, uint32_t hidden, uint32_t top_k) {
    return peer && peer->magic == STRATA_DS41_PREFILL_MAGIC &&
        peer->version == STRATA_DS41_PREFILL_VERSION && !peer->reserved &&
        (peer->flags & STRATA_DS41_PREFILL_REQUIRED_CAPS) ==
            STRATA_DS41_PREFILL_REQUIRED_CAPS &&
        tokens && tokens <= STRATA_DS41_PREFILL_MAX_TOKENS &&
        hidden && hidden <= 5120 && hidden % 32 == 0 &&
        top_k && top_k <= 16 &&
        peer->max_tokens >= tokens && peer->max_hidden >= hidden &&
        peer->max_top_k >= top_k && peer->max_routes >= tokens * top_k;
}

int strata_ds41_prefill_validate_request(
    const strata_ds41_prefill_request *r, uint32_t layers, uint32_t hidden,
    uint32_t context, uint64_t expected_sequence) {
    uint64_t bytes;
    if (!r || r->version != STRATA_DS41_PREFILL_VERSION ||
        r->sequence != expected_sequence || r->reserved ||
        !layers || layers > 256 || !hidden || hidden > 5120 || hidden % 32 ||
        !context || context > 131072 || r->hidden != hidden) return 0;
    if (r->op == STRATA_DS41_PREFILL_BEGIN || r->op == STRATA_DS41_PREFILL_END)
        return !r->layer && !r->first_token && !r->tokens && !r->routes &&
               !r->top_k && !r->activation && !r->activation_bytes;
    if (r->op != STRATA_DS41_PREFILL_EXECUTE &&
        r->op != STRATA_DS41_PREFILL_DECODE_RAW) return 0;
    if (r->layer >= layers || !r->tokens ||
        r->tokens > STRATA_DS41_PREFILL_MAX_TOKENS || !r->top_k || r->top_k > 16 ||
        r->first_token > context || r->tokens > context - r->first_token ||
        r->routes > r->tokens * r->top_k ||
        r->routes > STRATA_DS41_PREFILL_MAX_ROUTES) return 0;
    if (r->op == STRATA_DS41_PREFILL_EXECUTE) {
        if (r->activation != STRATA_DS41_PREFILL_Q8_1 &&
            r->activation != STRATA_DS41_PREFILL_Q8_D4 &&
            r->activation != STRATA_DS41_PREFILL_Q8_D2S6) return 0;
        /* 144 bytes per 128 values, retaining format-specific metadata. */
        bytes = (uint64_t)r->tokens * (hidden / 32) * 36;
    } else {
        if (r->tokens > 64 || r->activation != STRATA_DS41_PREFILL_F32) return 0;
        bytes = (uint64_t)r->tokens * hidden * 4;
    }
    return r->activation_bytes == bytes;
}

int strata_ds41_prefill_advance(
    strata_ds41_prefill_state *state, const strata_ds41_prefill_request *r,
    uint32_t layers, uint32_t hidden, uint32_t context) {
    if (!state || state->reserved || state->active > 1 ||
        state->sequence == UINT64_MAX ||
        !strata_ds41_prefill_validate_request(r, layers, hidden, context,
                                             state->sequence)) return 0;
    switch (r->op) {
    case STRATA_DS41_PREFILL_BEGIN:
        if (state->active) return 0;
        state->active = 1;
        break;
    case STRATA_DS41_PREFILL_END:
        if (!state->active) return 0;
        state->active = 0;
        break;
    case STRATA_DS41_PREFILL_EXECUTE:
        if (!state->active) return 0;
        break;
    case STRATA_DS41_PREFILL_DECODE_RAW:
        if (state->active) return 0;
        break;
    default: return 0;
    }
    ++state->sequence;
    return 1;
}

int strata_ds41_prefill_q8_repack(
    void *dst, size_t dst_bytes, const void *src, size_t src_bytes,
    uint32_t rows, uint32_t hidden, uint32_t padded_hidden, int to_wire) {
    uint64_t wire_bytes, native_bytes;
    uint32_t row, col, blocks, padded_blocks;
    unsigned char *out = (unsigned char *)dst;
    const unsigned char *in = (const unsigned char *)src;
    if (!dst || !src || dst == src || !rows || rows > 8192 ||
        !hidden || hidden > 5120 || hidden % 128 ||
        padded_hidden < hidden || padded_hidden > 5120 || padded_hidden % 512 ||
        (to_wire != 0 && to_wire != 1)) return 0;
    blocks = hidden / 128;
    padded_blocks = padded_hidden / 128;
    wire_bytes = (uint64_t)rows * blocks * 144;
    native_bytes = (uint64_t)rows * padded_blocks * 144;
    if ((to_wire && (dst_bytes < wire_bytes || src_bytes < native_bytes)) ||
        (!to_wire && (src_bytes < wire_bytes || dst_bytes < native_bytes))) return 0;
    if (!to_wire) memset(dst, 0, dst_bytes);
    for (row = 0; row < rows; ++row) {
        for (col = 0; col < blocks; ++col) {
            const size_t w = ((size_t)row * blocks + col) * 144;
            const size_t n = ((size_t)col * rows + row) * 144;
            memcpy(out + (to_wire ? w : n), in + (to_wire ? n : w), 144);
        }
    }
    return 1;
}
