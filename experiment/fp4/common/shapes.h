// MiMo-V2.6-Flash shapes (config.json of XiaomiMiMo/MiMo-V2.6-Flash-RL, checked 2026-10-11).
#pragma once

namespace fp4::mimo {
constexpr int kHidden = 4096;
constexpr int kExpertInter = 2048;   // moe_intermediate_size
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kLayers = 48;          // layer 0 dense (intermediate 16384), 1..47 MoE
constexpr int kMoeLayers = 47;
constexpr int kDenseInter = 16384;
constexpr int kVocab = 152576;
// Per expert, MXFP4: gate [2048 x 4096], up [2048 x 4096], down [4096 x 2048].
// One matrix = 8.39M weights * 4.25 bits = 4.456 MB; one expert = 13.37 MB;
// top-8 per MoE layer = 107 MB; 47 MoE layers = 5.03 GB of expert bytes per decoded token.
}  // namespace fp4::mimo
