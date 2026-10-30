#pragma once

#include "strata/host/glm_env.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::ds41 {
// Run before populating or pinning mappings: a read of a worker-owned tmpfs hole allocates its pages.
inline void validate_owned_stream_settings() {
    const char* stream = std::getenv("STRATA_DS41_OWN_STREAM");
    if (!stream || !*stream) return;
    const char* own = std::getenv("STRATA_DS41_REMOTE_OWN");
    const char* populate = std::getenv("STRATA_DS41_POPULATE");
    if (!own || !*own || !populate || std::string(populate) != "0" ||
        strata_env_value_double("STRATA_DS41_PIN_GB", 60.0) != 0.0 ||
        strata_env_is_one("STRATA_DS41_PIN_EXPERTS") || strata_env_is_one("STRATA_DS41_RAM_COPY"))
        throw std::invalid_argument("ds41: STRATA_DS41_OWN_STREAM requires STRATA_DS41_REMOTE_OWN, "
                                    "STRATA_DS41_POPULATE=0, STRATA_DS41_PIN_GB=0, "
                                    "STRATA_DS41_PIN_EXPERTS=0 and STRATA_DS41_RAM_COPY=0 (sparse expert shards)");
}
}  // namespace strata::ds41
