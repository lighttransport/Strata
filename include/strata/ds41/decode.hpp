// include/strata/ds41/decode.hpp - the DeepSeek-V4.1-Flash path of strata-glm-decode (src/program/ds41_decode.cpp).
#pragma once

#include <string>

namespace strata::ds41 {

/// True for a GGUF whose general.architecture is deepseek41 (llama.cpp PR #30817's schema).
bool is_ds41(const std::string& path);
/// strata-glm-decode's command line for such a model.
int run(int argc, char** argv);

}  // namespace strata::ds41
