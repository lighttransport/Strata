// include/strata/ds41/engram_wire.hpp - the Engram row reads a decoder without the tables sends to the node that holds the model.
// One request carries `count` page reads (shard number, byte offset in that shard file, length); the reply is their bytes
// in order. A request with count = 0xFFFFFFFF ends the service. Own ucomm context (the decoder is rank 0), own tags.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::ds41::engram_wire {

inline constexpr uint32_t kTagRequest = 30, kTagReply = 31;
inline constexpr uint32_t kMaxRowBytes = 16384;
struct Header {
    uint32_t count;
    uint32_t pad;
};
struct Entry {
    uint32_t shard;   // 1-based, the NNNNN of NAME-NNNNN-of-MMMMM.gguf
    uint32_t len;
    uint64_t offset;
};
inline constexpr size_t kMaxEntries = 8192;
inline constexpr size_t kMaxRequestBytes = sizeof(Header) + kMaxEntries * sizeof(Entry);

}  // namespace strata::ds41::engram_wire
