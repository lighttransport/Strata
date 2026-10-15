# Longer contexts: host-RAM KV cache, blocked indexer, VRAM pre-allocation (GLM-5.3-Flash)

Three opt-in switches that move the context limit of a single 16 GB card. All are off by default, and the
default path is unchanged.

| switch | what it does |
|---|---|
| `STRATA_GLM_KV_HOST=1` | the MLA latent caches (2 KiB per token per layer, 24.6 KB per token over 12 layers) live in mapped pinned host memory instead of VRAM |
| `STRATA_GLM_INDEX_BLOCK_POOLS=<n>` | prefill computes the indexer's per-head dot products in blocks of `n` pooled keys (4 tokens each; multiple of 64, at least 512) instead of over the whole context at once |
| `STRATA_GLM_VRAM_PREALLOC=1` | on first use of a GPU, take all free VRAM except `STRATA_GLM_VRAM_PREALLOC_MARGIN_MIB` (default 256) in one allocation and carve every buffer from it; the 1 GiB budget reserve and the free-memory headroom guard no longer apply. Set `gpu_budget_mib` to the card's size |

## Why the KV cache can leave the GPU cheaply

GLM-5.3-Flash uses sparse attention. For each new token and MLA layer, the indexer scores all pooled keys (128 bytes
per token per layer, which stay on the GPU), selects 512 pools of 4 consecutive tokens, and attention reads only
those 2,048 latent rows. Every GPU attention path first gathers the selected rows into a VRAM buffer
(`glm_mla_gather*`), so each row crosses PCIe once per query when the cache is in host memory:

- **decode / verify**: the gather kernels read the mapped host memory directly (zero copy), 2,048 rows x 2 KiB x
  12 layers = 48 MiB per query token;
- **prefill**: a chunk's queries together touch most of the cache, so before each MLA layer of a prefill chunk
  (at least `STRATA_GLM_KV_HOST_STAGE_MIN` tokens, default 64) the rows written so far are copied to a VRAM staging
  buffer with one DMA and the existing kernels read that. The staging buffer is released before the decode tier is
  sized, so decode gets the VRAM. If it cannot be allocated, prefill reads host memory directly (slower) and says so
  once (`KV_HOST staging unavailable`);
- new rows are written straight to host memory.

The VRAM this frees goes to the GPU-resident expert tier (about 3 expert slots per 1K tokens of context).

Not supported with host KV: several parallel sequences (refused), and the split-MLA option of two same-vendor GPUs,
which copies the whole cache to the second card (disabled, with a log line). Conversation snapshots copy the cache
through the host as before.

## Limits

- Host RAM: the pinned cache comes out of the page cache that holds the expert weights. On TR16 (125 GB) with
  REAP-224 (95.5 GB) a few GB are harmless; tens of GB push expert pages to disk and slow decode far more than the
  PCIe reads. REAP-50 (63 GB) leaves room for about 1M tokens.
- PCIe: TR16's RX 9070 XT reaches the CPU over Gen3 x16 (the card's own link is Gen5 x16 to a switch).
- The indexer still scans every pooled key for every token, and prefill grows quadratically with the prompt.

## Measurements

Pending: `tools/glm_kv_offload_probe.cu` (gather bandwidth from host memory), token-identity checks against the
default path, and 8K / 128K / 256K runs on the RX 9070 XT.
