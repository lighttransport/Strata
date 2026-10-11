// include/strata/host/glm_names.hpp - the C11 naming module, with the std::string the C++ call sites want.
//
// The rule stays in the C unit (include/strata/host/glm_names.h); this only decides how a C++ caller receives
// it.  A caller that writes into a fixed buffer itself should use the C header directly - these wrappers
// allocate, exactly like the `"blk." + std::to_string(l) + "."` expressions they replace.
#ifndef STRATA_HOST_GLM_NAMES_HPP_
#define STRATA_HOST_GLM_NAMES_HPP_

#include <stdexcept>
#include <string>

#include "strata/host/glm_names.h"

namespace strata::host {

/// `blk.<layer>.<suffix>`.  Throws rather than returning a truncated name: a short name resolves to nothing,
/// or worse, to something else.  `layer` must be a real one (0..n_layers-1); a negative layer throws too,
/// because the old `"blk." + std::to_string(l)` happily spelled it out and then failed a lookup with a name
/// that looked plausible.
inline std::string blk_name(int layer, const char *suffix) {
    char buffer[STRATA_NAME_MAX];
    if (strata_blk_name(buffer, sizeof buffer, layer, suffix) < 0)
        throw std::runtime_error("host: tensor name does not fit");
    return std::string(buffer);
}

/// `blk.<layer>.`
inline std::string blk_prefix(int layer) {
    char buffer[STRATA_NAME_MAX];
    if (strata_blk_prefix(buffer, sizeof buffer, layer) < 0)
        throw std::runtime_error("host: tensor name does not fit");
    return std::string(buffer);
}

/// True for a name this module would build.  Faster than comparing strings against a prefix, and it says the
/// same thing everywhere instead of at each `rfind("blk.", 0) != 0`.
inline bool is_blk(const std::string &name) { return strata_parse_blk(name.c_str(), nullptr, nullptr) != 0; }

}  // namespace strata::host

#endif // STRATA_HOST_GLM_NAMES_HPP_
