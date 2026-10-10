#pragma once

#include <cstddef>

namespace strata::kernels {

// GPU-driven decode step: the GPU publishes each MoE layer's routing and activation into mapped pinned
// memory and waits for the CPU's routed sum without a host round trip.
//
// All pointers are device views of one mapped pinned allocation. Flags use separate 128-byte lines:
// flags[slot * 32] is the published generation (GPU writes), flags[slot * 32 + 8] the generation whose resident
// rows are in the mailbox (GPU writes, canonical mode), flags[slot * 32 + 16] the completed generation (CPU writes). control[0] is the step generation (host writes before launch), control[16]
// an error code (GPU writes slot + 1 on timeout), control[32] an abort request (host writes).
struct GlmMailboxView {
    unsigned *flags = nullptr;
    unsigned *control = nullptr;
    int *ids = nullptr;        // [slot][max_tokens * top_k]
    float *weights = nullptr;  // [slot][max_tokens * top_k]
    float *act = nullptr;      // [slot][max_tokens * hidden]
    float *sum = nullptr;      // [slot][max_tokens * hidden]
    float *rows = nullptr;     // [slot][max_tokens * top_k * hidden], canonical mode only
    int *wanted = nullptr;     // [slot][max_tokens * top_k]: the router's unbiased choice, when requested
    int slots = 0, max_tokens = 0, top_k = 0, hidden = 0;
};

// Bytes of the mapped allocation described by the view geometry, and the view over a base pointer.
size_t glm_mailbox_bytes(int slots, int max_tokens, int top_k, int hidden, bool rows = false, bool wanted = false);
GlmMailboxView glm_mailbox_view(void *base, int slots, int max_tokens, int top_k, int hidden, bool rows = false,
                                bool wanted = false);

// Copies control[0] into *generation once per step; later kernels read the device copy.
void glm_mailbox_begin(GlmMailboxView v, unsigned *generation, void *stream);
// Writes the layer's IDs, weights and activations, then the published generation.
// `wanted` (optional, needs a view with a wanted region): the IDs an unbiased router would have selected.
void glm_mailbox_publish(GlmMailboxView v, int slot, const int *ids, const float *weights, const float *x,
                         int tokens, const unsigned *generation, void *stream, const int *wanted = nullptr);
// Waits until the CPU completed this generation, then out += sum. A timeout records an error and
// leaves out unchanged; an abort request returns at once.
void glm_mailbox_wait_add(GlmMailboxView v, int slot, float *out, int tokens, const unsigned *generation,
                          void *stream);
// As glm_mailbox_wait_add, for layers with GPU-resident experts: the CPU's sum covers only routes whose
// expert has no lookup entry, and out += sum + weights[j] * resident[j] over resident routes j in order.
void glm_mailbox_wait_add_resident(GlmMailboxView v, int slot, float *out, int tokens, const unsigned *generation,
                                   const int *ids, const float *weights, const unsigned long long *lookup,
                                   const float *resident, void *stream);

// Expert deferral (STRATA_GLM_DEFER_EXPERTS): out += weights[j] * resident[j] over resident routes, no wait.
void glm_mailbox_add_resident(GlmMailboxView v, float *out, int tokens, const int *ids, const float *weights,
                              const unsigned long long *lookup, const float *resident, void *stream);
// Expert deferral, one token: waits for the CPU's sum s of `slot`, then adds it to the four hyper-connection streams
// `r` as that layer's write would have: r[j] += w[j] * s with w[j] = earlier[4 + j] (the layer's write coefficients),
// or, when a later write with coefficients `later` already ran, w[j] = sum_i later[8 + 4i + j] * earlier[4 + i].
void glm_mailbox_wait_add_deferred(GlmMailboxView v, int slot, float *r, const float *earlier, const float *later,
                                   const unsigned *generation, void *stream);
// Canonical mode (canon_expert.hpp) with GPU-resident experts: copies the rows of resident routes (row j of
// `resident` for route j) into the mailbox rows and then sets the rows flag, flags[slot * 32 + 8]. The CPU adds
// every route in the canonical order and returns one sum, so the device reads back as little as without a tier.
void glm_mailbox_post_rows(GlmMailboxView v, int slot, const int *ids, const float *weights,
                           const unsigned long long *lookup, const float *resident, int tokens,
                           const unsigned *generation, void *stream);

} // namespace strata::kernels
