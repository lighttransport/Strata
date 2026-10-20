#pragma once
#include "strata/kernels/cpu/pool.hpp"

namespace strata::kernels::cpu::detail {
// Measured original-layout FP4 worker route. Keep each call independent so a
// smaller route or a different verification width cannot inherit six tasks.
inline int nvfp4_split_tasks(bool enabled, int base, int explicit_tasks,
                            int participants, bool owned, const NativeFmt& f,
                            const ExpertJobMulti* jobs, int count) {
    if (!enabled || base != 3 || explicit_tasks || participants != 4 || owned ||
        (count != 2 && count != 3) || !jobs || f.n_embd != 4096 || f.n_ff != 2048 ||
        f.gu_type != kNvfp4Type || f.d_type != kNvfp4Type ||
        f.nvfp4_k32 || f.canon || f.gate_skip != 0 || f.swiglu_limit != 10 ||
        f.fuse_h_quant || f.observer) return base;
    const int nt = jobs[0].nt;
    if (nt != 1 && nt != 2) return base;
    if (count == 2 && nt != 1) return base;
    for (int i = 1; i < count; ++i) if (jobs[i].nt != nt) return base;
    return 6;
}

// Opt-in measured worker geometry only. Nondefault per-participant counts and
// fixed task budgets win. Recompute per call; small routes cannot inherit twelve.
inline int q23_split_tasks(bool enabled, int base, int explicit_tasks,
                           int participants, bool owned, const NativeFmt& f,
                           const ExpertJobMulti* jobs, int count) {
    const bool measured_shape = f.n_embd == 4096 &&
        ((f.n_ff == 2048 && (count == 3 || count == 4 || count == 8)) ||
         (f.n_ff == 1024 && count == 8));
    if (!enabled || base != 3 || explicit_tasks || participants != 4 || owned ||
        !measured_shape || !jobs ||
        f.gu_type != 10 || f.d_type != 11 ||
        f.canon || f.gate_skip != 0 || f.swiglu_limit != 10 || f.q3_expanded_down ||
        f.fuse_h_quant || f.observer || f.hidden_transform) return base;
    for (int i = 0; i < count; ++i) if (jobs[i].nt != 1) return base;
    return 12;
}
}
