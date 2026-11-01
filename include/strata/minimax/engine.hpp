// include/strata/minimax/engine.hpp - the MiniMax M3 / M3.1 Flash (GGUF architecture minimax-m3) engine path as a
// library: `strata` hands a minimax-m3 model to it at the top of main() (the family switch), and strata-m3 is a thin
// wrapper around the same entry point.  See src/minimax/m3_engine.cu.
#pragma once

namespace strata::minimax {

/// The engine's command line: --model / --native GGUF (any shard of a split set), --tokens IDS (comma or space
/// separated), --max-new N, --max-context N, --chunk N, --expert-vram GiB, --dump-logits FILE.
/// Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::minimax
