# Strata on two or more GPUs (layer split)

For the experimental **GLM 5.3f** decoder, see [GLM_MULTI_GPU.md](GLM_MULTI_GPU.md).
It uses separate `gpu_layout=layer` settings and a one-ahead MTP scheduler.
The setup instructions below describe the Qwen engine.

One model can run across several NVIDIA cards in one PC. The layers are split into contiguous ranges, one per GPU:
the first card runs layers 0 to K-1, the next card runs K onward, and so on; the last card also runs the output head
and the draft (MTP) layer. Each card keeps an expert cache for **its own layers only**, so two cards hold about twice
the experts one card holds - for the Coder model on a 16 GB + 24 GB pair, nearly all of them, which is where the
speed comes from (decode then barely touches the CPU pool).

This is pipeline (layer) parallelism, not tensor parallelism: a token crosses from one card to the next once per
verify window (a few hundred KB through pinned RAM), not twice per layer. No NVLink or peer-to-peer access is
needed; cards on x4 or x1 slots work, and the PCIe share of each card is probed on its own link. A `--pcie-frac` you give
is every card's share and skips those probes; there is no per-card setting yet.

## Using it

**Nothing to type.** `START-HERE.bat` (Linux: `./setup.sh`) lists your NVIDIA cards and says for each one whether
Strata can use it:

```
  Your NVIDIA GPUs:
    GPU 0: NVIDIA GeForce RTX 5080, 16 GB VRAM - can be used
    GPU 1: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - not supported - older than the RTX 20 series (compute capability 6.1; Strata needs 7.5 or newer)
    GPU 2: NVIDIA GeForce RTX 3090, 24 GB VRAM - can be used
  ...
  1) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) + GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) together   (recommended)
  2) GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) only
  3) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) only
Which GPUs? [1]:
```

When two or more cards can share the model, the two best together are recommended (the newest generation first:
it becomes the main card). A model installed on one card asks once, at its next start, whether to use both from
now on; the answer is kept.

**Choosing yourself** (at setup or at any start):

```
--gpus 0,2                 these cards together, as nvidia-smi numbers them; the first is the main one. Remembered.
--gpus all                 every card that can share the model
--gpu 0                    one card (at a start: for that start only)
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

**Not supported** (setup says so and names the cards that can be used instead):
- a card older than the RTX 20 series (compute capability below 7.5: GTX 10 and older), unless you name it: Pascal
  and Volta cards (Tesla P100 / P40, GTX 10, V100) are admitted when you choose them with `--gpus` (or `--gpu`,
  `--cuda 12`) and then run the experimental CUDA 12 engine, see [OLDER_GPUS.md](OLDER_GPUS.md). Two Tesla P40s
  (`"gpu": [0, 1]` in the config) ran the layer split in a community benchmark: IQ2_XS decode 19.6 tok/s on one card,
  34.8 on both (#1028, experimental, one report);
- a card with less than 8 GB of VRAM, together with others (each card holds a copy of the dense weights and its
  own prompt buffers) - unless you name it with `--gpus`: then setup says the risk and asks (`--yes` with the named
  cards goes ahead);
- Intel GPUs, and a mix of NVIDIA and AMD cards. (AMD cards share a model among themselves: `./setup.sh --backend
  hip --gpus 1,0`, see [AMD_HIP.md](AMD_HIP.md).)

Or edit an existing config (`strata-*.json`), then restart:

```json
"gpu": [0, 2],
"layer_split": "auto"
```

`"layer_split"` is `"auto"` (placed by each card's free VRAM) or the **first layer of each later card**: one rising
number per card after the first, not a count of layers per card. With 4 cards and a 48-layer model, `"24,36,42"`
(or `[24, 36, 42]`) puts layers 0-23 on the first card, 24-35 on the second, 36-41 on the third and 42-47 on the last.
The server checks it before the start and says what is wrong (0.1.39, #644).

**Card order with `"auto"`** (NVIDIA, #1352): the faster card (multiprocessors x max clock) goes **last** - the last stage runs the head, the draft layer and the verify, and a prompt chunk waits on it (a 4070 Ti SUPER + 5060 Ti read a 6K prompt in 50 s one way round and 15 s the other). Cards whose scores are within 5% of each other (two of the same model) keep your order; a manual split keeps it too. `"gpu_order": "as_given"` keeps the order you wrote under `"auto"` as well. The engine log line `layer split: card order ...` says when it changed.

**Skip the split when the first card holds everything** (opt-in, 0.1.31): `"split_skip_if_fits": true` in the config
(engine flag `--split-skip-if-fits`, with `--layer-split auto`) runs on the first card alone when it holds every
profiled expert plus the context's KV, the draft layer and the reserve, and says so in the log; otherwise the split
stays. On an R9700 32 GB + RX 9070 XT the R9700 holds all of the Coder's experts: with the flag 4K prompts read at
1,776 tok/s instead of 1,244 (split) and decode runs at ~60 tok/s instead of ~51 (16K prompts ~5% slower than split).

**Short prompts on a split (0.1.32, #340).** 0.1.30 gave each card's prompt path a loan from its own expert cache,
refilled after every request; on cards that hold nearly all their experts that cost short prompts up to a third of
their speed. 0.1.32 refills all cards at once, uses a smaller streaming ring on a split, and lets a card with free
VRAM keep its own prompt buffers - the same output as 0.1.31, measured on an R9700 + RX 9070 XT: 2K prompts 993 ->
1,265 tok/s, 16K 1,852 -> 1,950, decode unchanged. `STRATA_SPLIT_OWN=1` (opt-in) gives every card its own buffers:
2K 1,450 and 16K 2,227 tok/s there, but a full card then keeps a different set of experts resident, so the output
differs from the default's (stable and coherent); `STRATA_SPLIT_OWN=auto` does that only where the buffers are at
most 12% of each card's VRAM.

**The idle card can help one-chunk prompts (opt-in, `STRATA_PREFILL_HELP=1`).** A prompt that fits one chunk runs the stages one after the other, so while
one card reads its layers the other idles. With it on, each stage hands a share of its streamed experts to the idle card: it
streams them over its own PCIe link into its own (lent) prompt buffers, computes their rows on the MMQ path and sends
them back - `--peer-device`'s peer streaming, without P2P (the activations and the rows go through mapped host memory,
read by copy kernels, so they do not queue behind the expert blobs on either card's copy engine). The share falls with
the prompt (0.41 of the streamed experts at 1.5K tokens, 0.32 at 3K) and is off from ~3.3K tokens, where the stages
overlap anyway and no share paid. Measured on 2x RTX 3090 (UD-Q4_K_XL, no P2P), prompt tok/s without / with: 1.1K 496 /
723, 1.5K 674 / 833, 2K 878 / 1,128, 2.5K 1,017 / 1,296, 3K 1,260 / 1,440; 4K and 8K unchanged, decode unchanged. It
costs no VRAM (the idle card's own prompt buffers) and ~110 KB of mapped host memory per token of the largest chunk it
helped (~360 MB at 3.3K tokens). The rows it computes round like a different MMQ grouping, so the output is not
bit-identical to the default's (it is repeatable: same prompt, same output), which is why it is **opt-in**:
`STRATA_PREFILL_HELP=1` turns it on. Native packs on the MMQ prompt path only (not with the fused prompt kernels,
`STRATA_PF_FUSED=1`, nor with `--peer-device`). With it on, `STRATA_PREFILL_HELP_FRAC=f` fixes the share.

The engine flags behind it: `--layer-split K1[,K2..]|auto` and `--split-device D1[,D2..]` (the later stages'
devices; default the next visible ones). `--layer-split K --split-device 0` runs both stages on one card sharing
everything - the bit-exact check of the hand-off, not a speed mode.

**The draft layer's prompt K/V in batches on a split (opt-in, `STRATA_SPLIT_MTP_BATCH=1`).** With `--mtp` the draft
layer sits on the last card, and each prompt chunk ends by filling that layer's K/V for the chunk's rows. On one card the
prompt path does it in batches (`Prefill::draft_kv`, a few large matrix products per chunk); on a split the drafter's own
pass always did it, one graph launch and four device copies per group of 8 rows (1,024 groups per 8,192-token chunk).
With the variable set, the last card's prompt path batches it as it does on one card. Restart-only; default off, so the
default start is unchanged. It needs a `--native` pack (the GGUF-form token table) and a paged or ring (`--kv-resident`)
draft K/V (`STRATA_MTP_BATCH_RING=0` sends a ring back to the old pass); otherwise it says why once and keeps the old pass:
`strata serve: draft layer prompt K/V batched on CUDA<d> ...` or `strata serve: STRATA_SPLIT_MTP_BATCH=1 declined, the
drafter's own pass runs: <reason>`. The draft layer's K/V come out of Q8_1 x Q8_0 MMQ instead of its own mmvq, so the
drafts, and how many are accepted, can move; the target's tokens are decided by the verify window.
`tools/split_mtp_batch_parity.py` compares the greedy texts of the two arms. Measured on one machine (2x RTX 3080 20 GB
at 220 W, Xeon E5-2696 v4, UD-Q4_K_XL, `--layer-split 23`, `--batch-mtp`, two slots, `--prefill auto:16384`), two full
restarts per arm in the order off, on, off, on, the same binary in every arm: pooled medians of the prompt read were 1,996
tok/s without and 2,129 with at 25K tokens (+6.7%), 2,448 and 2,553 at 51K (+4.3%), 2,846 and 2,940 at 104K (+3.3%); all 8
on/off ratios of medians are above 1 (1.03 to 1.08), and the two off restarts differ by at most 2% at 25K and 0.3% at the
other sizes. The host's "after each chunk" time per request fell from 0.9-1.3 s to 0.1-0.25 s (one restart per arm). Greedy
output (`--adapt-every 0 --suffix-draft 0`, five prompts of 256 tokens and one 25.5K-token prompt) was identical in 6 of 6
prompts in all five comparisons, the two runs inside one process (A/A) included, and the drafts accepted were 826 of 1,210
in every run, so no difference in acceptance was visible. Decode is not shown to be unchanged: the medians with the variable
set were about 3% lower (pooled solo 82.6 -> 80.2 tok/s, two streams 93.2 -> 90.8), and the same configuration restarted
twice moved up to 4.7%, so two restarts per arm can neither confirm nor exclude it; the change touches only the prompt path.

**Each card loads only its own layers' dense weights** (0.1.39, PR #639) with explicit split points (`--layer-split
27`, not `auto`): every card used to keep a full copy (~3.4 GB for the Coder) though its stage reads only its own
layers, and the VRAM it frees goes to that card's expert cache (2x MI50 16 GB, Coder: 8,819 -> 10,626 experts in
VRAM, decode 39.2 -> 41.7 tok/s). Opt-in for now, on AMD and NVIDIA alike: `STRATA_STAGE_TRIM=1` (please report how
it goes).
A card holding more experts can change which experts run on the GPU, so the output can differ slightly from a run
without it.

**The resident RAM mode works on a split** (`--resident-experts`, the low-RAM mode). The RAM copy of the experts
leaves out the ones every card's cache holds, not only the first card's, and when the rest does not fit whole it keeps
the hottest by the expert profile over all the layers. An adaptive swap copies the evicted expert back into RAM from
the card that owns its layer. Before, `--resident-experts` with a split ran as `--mmap-experts`, and setup recommended
one card in the low-RAM mode; with an engine that has it (`RESIDENT_SPLIT_ENGINE` in setup.py), setup keeps the cards
together and the experts no card holds in RAM. The same copy, sized by a RAM budget (`--resident-budget-gib`, the
Unsloth UD-Q4_K_XL and UD-IQ4_XS configs), is kept on a split as well ([UNSLOTH_Q4.md](UNSLOTH_Q4.md)). Swift 1.5 IQ3_XXS at
160K (q4_0 KV, `--prefill 4096`, `--spec 4` with the stock draft layer), RTX 4060 Ti (layers 0-19) + RTX 5080 (20-47),
i9-14900KF, 32 GB of RAM, Windows 11, four greedy prompts at a time, decode tok/s:

| | first four prompts | after three more rounds |
|---|---|---|
| 5080 alone, `--resident-experts` | 25 | 29 |
| split, `--mmap-experts` (what `--resident-experts` became on a split) | 32 | 64 |
| split, `--resident-experts` (22 GiB of experts locked in RAM) | 71 | 69 |

The split with `--mmap-experts` catches up once the OS file cache holds the experts, on a PC with nothing else
running; the resident copy is there from the first request and stays locked when other programs need the RAM. With
`--pcie-frac 0 --adapt-every 0` the split's greedy output is the same with either mode.

**The prompt path's loan on a split and its streamed ring (`STRATA_SPLIT_RING`).** Under a split the prompt path borrows
the tail slots of each card's expert cache for its buffers (they are given back after the prompt), and the resident RAM
copy keeps those experts too, so a borrowed expert is not read back from the model file during a prompt. How many slots
are borrowed depends on the prompt path's streamed ring: a split uses 96 ring slots when at least 75% of the (layer,
expert) pairs sit in some card's cache, and `STRATA_SPLIT_RING=N` sets N slots (`0` goes back to the pinned-share rule).
The ring used to be chosen after the RAM copy's regions were sized, so the regions were sized for a different ring than
the prompt path then used: with `STRATA_SPLIT_RING=384` it borrowed 5,396 slots and the copy kept 4,662, and the
other 734 were read from the model file in every chunk. The ring is now chosen first (`Prefill::set_ring_override`
says it must be set before the buffers are counted). The start-up line `strata serve: lend sizing: the prompt path
borrows N slots (ring R slots at chunk C), K of them keep their experts in RAM too` shows both numbers, and a
`WARNING` follows when K is below N. Without a split, or without `--resident-experts`, nothing changes. With a split
the order of the two steps changed for every ring choice, the 96-slot rule included: its regions used to be sized for
the larger default ring, so there the copy probably kept more slots than were borrowed (not measured). The 5,396 /
4,662 figures were measured on the test machine (2x RTX 3080 20 GB at 220 W, Xeon E5-2696 v4, UD-Q4_K_XL,
`--layer-split 23`, resident RAM mode, `--prefill auto:16384`, 2 slots), one restart per arm, with `STRATA_SPLIT_RING=384`
and `STRATA_PREFILL_RING` unset. Before the change a 104K-token prompt made 413 blob reads and read 10.3 GB of experts
from the model file per request; after it, 0, with all 5,396 borrowed slots kept in RAM, for 2.15 GiB more pinned RAM
(13.61 -> 15.76 GiB). 25K and 51K prompts read nothing from the file in either case. With the override unset and
`STRATA_PREFILL_RING=384` the lend lines are identical before and after (5,396 / 5,396, 15.76 GiB). No speed gain is
shown: the medians of three 104K reads were 2,803 tok/s before and 2,907 after, but the spread before is 2,540-2,897
(its first read faulted 4.2 GB in from NVMe) and there is no A/A restart. Here the page cache absorbed the 10 GB per
request; with less spare page cache it would be disk traffic. Not measured: both ring variables unset.

**A separate VRAM reserve for the later cards:** `--vram-reserve-later-mib N` (default: `--vram-reserve-mib`'s value).
The card that drives the monitors needs more headroom than one that drives none; with the display on the last card,
`--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first card's cache that VRAM.

**auto** tries every placement (all of them for two, three or, since 0.1.40, four cards; proportional to the free VRAM beyond that) and
keeps the one whose caches would hold the most of the expert profile, hottest pairs weighted most; ties go to the
placement that leaves the fullest card the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

## What each card holds

- **every card**: a copy of the dense weights (~3.4 GB for the Coder), its own session state (the KV cache of the full
  context), its verify window and its prompt-path buffers, and an expert cache for its layers filled from the profile;
- **the last card**: also the output head and the draft layer (~0.8 GB);
- **host RAM**: the expert arena once, shared by all cards (the CPU pool computes whatever no card holds).

Prompts are read in chunks that flow through the cards in turn; while a later card reads chunk c, the first card
already reads chunk c+1. (0.1.44, AMD gfx12: the auto chunk on a split stays at 8192, not the single-card 16384: a 16K prompt is then two chunks
that overlap instead of one, a 32K one four instead of two. Same tokens; 2x / 3x / 4x R9700 prompt tok/s at 16K +36% / +45% / +53%,
32K +18% / +21% / +29%, 64K +4% / +12% / +17%, 8K unchanged. `STRATA_SPLIT_CHUNK16K=1` keeps 16384.) Conversation checkpoints save and restore every card's state; the adaptive expert swaps copy
into the card that owns the layer.

## Limits (for now)

- **Works across cards** (bench/results/2026-09-29-layer-split-limits):
  - images (`--vision`): each card keeps its own image-position table;
  - control vectors and the experimental speed projection: each card holds the vector's tables, switched on and
    off per request on all of them;
  - KV streaming (`--kv-resident`): each card streams the KV of its own session;
  - mid-prompt checkpoints (`--prompt-cache-every`): each card saves its part of a checkpoint when it has read that
    chunk;
  - the older helper-GPU caches (`--expert-cache-remote`, docs/SECOND_GPU.md): they take the visible GPUs no stage
    runs on, and hold only experts no stage's cache holds. On the test rig, a 2080 Ti helper made decoding slower,
    as it did without a split: its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`), with or without a split; a native (IQ) pack says so at
  start.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024`
  halves that) instead of borrowing cache slots as one card does. An explicit `--expert-cache` on the first card is
  capped to leave room for them.
- Under WDDM (Windows, and WSL2) only 8 GiB of the expert arena is pinned (more, mapped into two GPU contexts,
  leaves WDDM refusing allocations); the rest streams through the pinned staging ring. A Linux driver has no such
  limit, so there the whole arena is pinned (since 0.1.31; the cap cost a 4090 + 3060 split two thirds of its
  prompt speed, #253). `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on any OS.
- Every card needs compute capability 7.5 (RTX 20 or newer). The pre-sm_80 QSA scorer path is fp32 FMAs, so a
  Turing card runs the same kernels instead of the tensor-core prompt attention.

## Measured

The Coder on an RTX 5080 + RTX 3090 (Ryzen 9 9950X3D), 32K context; details in
`bench/results/2026-09-29-layer-split/`:

| | Prompt 16K / 28K tok/s | Decode story / code tok/s |
|---|---|---|
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

- **Prompts gain the most** (+18-20%): each card reads its own layers of the chunk while the other reads the next.
- **Decode is on par with the faster card alone**, and ahead on code. Once both caches hold nearly every routed
  expert, the per-layer GPU time decides.
- **Correctness:** one GPU is byte-identical to 0.1.20, and the hand-off itself is bit-exact.

**Which cards and in what order:**
- Put the fastest card first; auto gives it as many layers as its cache allows.
- Leave out a much slower card when two already hold the model. An RTX 2080 Ti as a third card made the 5080 +
  3090 pair slower (68 / 90 tok/s decode): every extra card costs its own round per window.
- More cards pay off when the model's routed experts do not fit the faster ones.

## One conversation with both cards busy (`--pipeline-windows`, default on two or more AMD gfx12 stages)

**Default on AMD RDNA4 (gfx1200 / gfx1201).** With `--serve`, the draft layer and a layer split of **two or more stages**,
`--pipeline-windows 2` is on unless you say otherwise (the log says "is on by default for a layer split of N stages"; two
stages since 0.1.44, three or more since 0.1.43). It is no quality trade: the tokens are the serial loop's up to CPU rounding
(below). `--pipeline-windows 0` or `STRATA_PIPELINE_AUTO=0` turns it off; `--pipeline-windows 2` asks for it on any split.
The same options that switch it off when you ask for it (`--batch`, `--peer-device`, the helper caches, no draft layer,
`--adapt-async 1`) leave it off without a line when it is only the default. NVIDIA builds keep it opt-in.

**Where the draft layer sits (0.1.44).** The draft layer's chain used to run on the last card, beside the last stage's window.
On 2x / 4x R9700 the two streams then slowed each other, in most engine starts and for the whole life of that engine: the
last stage's window took 27 ms instead of 13 ms while a chain ran (14.9 ms in the lucky starts, which also decoded 25% faster;
the 0.1.43 notes' "128.9 does not reproduce" was this). With the pipeline on, the engine now loads the draft layer on the
**first** card, whose window has finished when the chain starts: the chain's source rows are copied across once per chain and the
draft head's rows once at load. Same tokens; `STRATA_MTP_STAGE0=0` keeps the old place. Exact-flags A/B, 5 pairs, decode tok/s
(serial loop -> now): 2 cards code 103 -> 125 (+21%), 4K prompt +8%, prose equal; 3 cards code +19%, 4K +12%, prose +2%; 4 cards
code +22%, prose +4%, 4K +15%.

With a split the cards take turns on a verify window: the first card runs its layers and hands off, then waits while
the last card runs the rest, the head and the draft. `--pipeline-windows 2` lets the first card start the **next**
window while the last card still verifies this one. The next window is a guess: that this window is accepted whole
and that its bonus token is the one the draft layer predicts (the draft layer is run on through the drafts of the
window in flight). When the guess holds, half of the next window is already done; when it does not, the first card
puts its state back (a copy of its recurrent state taken while the window ran) and the next window is built from the
real tokens. A guessed window is only started when the draft layer's estimate says it is likely to be kept. Windows
copied from earlier context (`--suffix-draft`) are guessed past as well, which is where edits that copy text gain
most. `--pipeline-windows 1` overlaps only the short prompt reads that go through the verify windows
(`--short-read`).

Two or more cards, `--serve`. With three or more stages every stage but the last is a front stage: the guessed
window follows the verified one through them one card behind (each front stage keeps its own pair of recurrent-state
copies and puts its state back on a wrong guess), and only the last stage waits for the verdict. `--pipeline-windows 1`
and the asynchronous tier stay two-stage. In the config:

```
"args": [ ..., "--pipeline-windows", "2" ],
"layer_split": "20"
```

- **Cost**: a second verify window on each card, 160 MiB more kept out of each card's expert cache, plus two copies
  of the first card's recurrent state (about 3 MiB per GDN layer it runs) on the first card with `2` (on every front
  card with three or more stages; only the first card's room is kept out of its cache, the others allocate theirs after).
- **Same text**: the last card only ever runs windows that are verified, and every window row computes what it
  would in any other window, so the tokens are the serial loop's. With `STRATA_IQ_MT_MIN=1 --pcie-frac 0
  --adapt-every 0` the greedy output is identical bit for bit to the serial loop's with the same expert caches. The pipeline keeps
  its VRAM out of the caches, so against a run without the flag a few experts move from a card to the CPU, which
  rounds them differently, and a near-tie can flip (a serial run given the same caches through `--vram-reserve-mib`
  matches it exactly).
- **Off, with one line in the log saying why**, with `--batch` slots, `--peer-device`, the helper caches
  (`--expert-cache-device1..3`, which `--remote-expert-opt` builds on), a split onto one GPU (`--split-device 0`), or no
  draft layer; on three or more stages also `--pipeline-windows 1` and `--adapt-async 1`. A request with repetition penalties (`penalty_last_n`) or coupled
  draft sampling decodes serially.
- **With the resident RAM mode's asynchronous swaps** (`--adapt-async 1`, [DETAILS.md](DETAILS.md)) a round's steps
  advance between the verified windows. Each card's copies are queued by the decode loop itself while that card has
  no window in flight, after every window that may still read what they overwrite has finished; the moves into RAM
  wait the same way.
- **Measured** (Swift 1.5 IQ3_XXS, 160K context, q4_0 KV, the stock draft layer, RTX 4060 Ti (layers 0-19) +
  RTX 5080 (20-47), i9-14900KF, 32 GB of RAM with the resident RAM mode on the split (#848); greedy, 500 tokens, two
  interleaved pairs of three rounds, decode tok/s): Python code 90.4 -> 103.1, C code 76.6 -> 81.6, English prose
  63.0 -> 71.1, Italian prose 41.6 -> 46.3, a copy-heavy edit (a 5 KB file back with a rename) 90.5 -> 117.9; mean
  72.4 -> 84.0 (+16%). It pays when the windows are GPU-bound: with the experts read through the OS file cache
  (`--mmap-experts` on that 32 GB PC) the file reads dominate and it measured no faster.
- **Measured on four stages** (Flash-Next GSQ-RCO IQ3_S, 262K context, int8 KV, layer split 12,24,36 on 4x RX 7900 XT,
  every expert in VRAM; greedy, reasoning off, 256 tokens, repository-text prompts): decode 59.4 -> 67.6 tok/s at 4K
  (median of 3) and 51.8 -> 65.6 at 32K. A guessed window that holds costs ~15 ms against ~43 ms for a fresh one; about
  a third of the guesses held. The greedy text was identical to the serial loop's on all four prompts (`STRATA_IQ_MT_MIN=1
  --pcie-frac 0 --adapt-every 0`), and with every guess forced wrong (`STRATA_PIPELINE_FORCE_MISS=1`).
- **HIP**: the draft layer's per-row `gr_read` used the last stage's session scratch (`ss.block.gr`), which that stage's
  verifier also uses; a chain launched at a verdict and the next window's last stage then ran on the card at once and
  the window's head read a clobbered mix (wrong tokens after a guess that held). The draft layer now has its own scratch.

The `STRATA_PIPELINE_*` tuning and test variables (THETA, FORCE_MISS, SWITCH, LOG, TRACE, SPEC_DEPTH and the like) are read only with
`STRATA_PIPELINE_DEBUG=1`. `--pipeline-windows` and `--adapt-async 1` combine: the engine turns the asynchronous tier
off beside `--pipeline-windows 2` only when `STRATA_PIPELINE_ADAPT_ASYNC=0` is set. What switches either one off is
printed once at start ("is off: ..."). `--remote-expert-opt` does something only with a helper cache
(`--expert-cache-device1..3`); on a plain layer split it is inert, and setup no longer writes it there (#1447).

Measured on 2x RTX 3090 (sm_86, 250 W limits; GPU0 PCIe 4.0 x16, GPU1 x4, no NVLink; Ryzen 7 9800X3D, 32 GB RAM),
Qwen3.8-Flash-Next GSQ-RCO IQ3_S, `--resident-experts`, KV int8, `--spec 4 --mtp`, `"layer_split": "29"` for the
`--pipeline-windows` rows (without `--remote-expert-opt`); the others are setup's config (`auto`). Decode is the mean of
runs 2-6 of 1,500-token coding replies at temperature 0.6, prefill one cold 19.9K-token prompt; one run per arm unless
noted (reported by adambenhassen, #1447; not repeated on our boxes):

| Arm | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| setup defaults (two runs) | 137.0 / 136.9 | 1798 |
| `--adapt-async 1` (two runs) | 145.2 / 143.7 | 1800 / 1797 |
| `--adapt-async 1`, no `--remote-expert-opt` | 145.0 | 1807 |
| `STRATA_ADAPT_LAG=2` | 142.8 | 1800 |
| `STRATA_EXCHANGE_ROTATE=1` | 139.6 | 1800 |
| `STRATA_PF_FUSED=1` | 137.2 | 1913 |
| `--pipeline-windows 2` | 136.7 | 2097 |
| `STRATA_SPEC_COUPLED=1` | 137.6 | 1804 |
| `STRATA_SPEC_PROB=1` | 137.2 | 1788 |
| `STRATA_SPEC_COUPLED=1` + `STRATA_SPEC_GUMBEL=1` | 136.5 | 1805 |
| async + lag 2 + rotate + pf_fused | 145.0 | 1829 |
| pw2 + lag 2 + rotate + pf_fused | 135.9 | 2296 |
| pw2 + async + lag 2 + rotate + pf_fused | 142.0 | 2302 |

The last row keeps most of the asynchronous tier's decode gain and the pipeline's +28% prefill; no stalls in any arm.
These are opt-in settings on one rig, not defaults.

## Several conversations at once

With a layer split, `--batch N --batch-groups G --trim-stage-weights` decodes several conversations together and
pipelines them through the cards: see [BATCHING.md](BATCHING.md).

## Experimental GLM runner with CUDA and HIP on Linux

`tools/glm_mixed_run.py` connects two separately built executables: the primary
`strata-glm-decode` and the other backend's `strata-glm-gpu-worker`. Each process
links one vendor runtime. The primary owns attention, recurrent state, routing,
sampling, and MTP. The worker computes whole experts and returns FP32 rows.
The existing single-GPU command remains the default. This experiment does not
add installer or server support.

Build `strata-glm-decode` and `strata-glm-gpu-worker` in both the CUDA and HIP
build directories. For ROCm 10.1, the launcher sets
`LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib` before invoking either executable,
including its protocol-size query. See [AMD_HIP.md](AMD_HIP.md) for HIP build
settings.

The experimental profile is
[glm53f-q23-mixed.json](../configs/experimental/glm53f-q23-mixed.json).
It requires the assembled GLM Q2/Q3 GGUF, including its retained IQ layers,
and FP16 tensor expert prefill. Worker decode uses the same canonical Q2/Q3
executor as primary GPU residents. Retained IQ uses the existing native GPU
executor. Canonical reduction follows route order. Routing and MTP acceptance
settings must stay fixed across single and mixed comparisons.

Example with HIP primary and CUDA worker:

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
export STRATA_GLM_EXPERT_PRIOR=build-q2-redesign/calibration/coverage.json
export STRATA_GLM_MIXED_PRIMARY_FRACTION=0.5
python3 tools/glm_mixed_run.py \
  --primary build-hip-rocm101-tr16/strata-glm-decode \
  --worker build-glm-release/strata-glm-gpu-worker \
  --model models/GLM-5.3-Flash-Q23-assembled.gguf \
  --worker-cache-mib 12288 --worker-gpu-budget-mib 14336 \
  --worker-numa-node 1 --report runs/mixed/memory.json -- \
  @docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids \
  512 4096 15 --context=12288 --prefill-batch=8192 \
  --gpu-budget-mib=15360 --decode-experts=cpu --decode-cache-mib=5632 \
  --prefill-experts=f16-batched --speculative=mtp --draft-depth=1
```

TR16 has the RX 9070 XT on NUMA node 0 and RTX 5060 Ti on node 1. Node numbers
in that example are machine-specific. Bind the worker before it allocates its
bounded pinned buffers. Model weights remain read-only mappings of the same
GGUF, so their physical file pages can be shared. The shared request mapping is
about 136.8 MiB; prefill transfers at most 8192 activation tokens once per
layer request. Outputs stream in chunks of at most 512 FP32 expert rows. Each
chunk requires an acknowledgement before the worker reuses the output buffer. No cross-vendor device pointers or peer copies are used.

The worker budget includes a 1 GiB runtime reserve. The default cache request is
12 GiB; requests up to 14 GiB are capped by observed free VRAM and runtime headroom and reduced to fit observed free VRAM. Allocation charges are probed
for each expert size; a further 512 MiB covers lazy library workspace. Decode ownership follows the calibration prior, with a bounded number
of slots per layer. Primary residents take precedence; worker-owned experts
are excluded from primary adaptive admission. CPU misses use the primary's
existing pool. Worker cache entries use recency for bounded eviction. Prefill
partitions 16-expert groups by the requested primary fraction. The worker uses four-expert FP16 GEMM panels, with row buckets rounded to 32
and 128 rows per expert per panel by default.
`STRATA_GLM_MIXED_PREFILL_TILE=256|512` selects larger bounded panels; returned
chunks still contain at most 512 rows. Gate, up and down weights are
dequantized once per group and reused across its panels. Buffers remain bounded.

Requests validate protocol version, geometry, expert IDs, destination rows,
sequence, and ownership generation. There is one outstanding buffer lease.
MTP requests carry no worker recurrent state. A stale reply or worker failure
aborts the run, and the launcher terminates the processes it created. Model
setup may take longer than two minutes; an idle worker has no time limit, while
an outstanding reply has a two-minute timeout.

### Validation and measurement

Host tests `glm_mixed_test` and `glm_mixed_transport_test` cover bounds, duplicate
rows, ownership, shared mapping transport, stale replies, and disconnects. The
transport test needs local Unix socket access, which may be blocked inside an
agent sandbox. Set `STRATA_GLM_MIXED_CHECK_ROWS=1` for diagnostic CPU/GPU row
checks. Do not include that mode or task tracing in throughput measurements.

`strata-glm-gpu-worker --transfer-bench` measures pinned H2D and D2H transfers
from 4 KiB to 32 MiB. Run the two backends separately and concurrently. Use
external CUDA/ROCm profilers in separate runs from throughput timing.

For fresh single-GPU controls, both mixed orientations, and prefill fractions
0.25, 0.5, 0.75, and a fraction derived from fresh single-GPU prefill rates:

```bash
systemd-run --user --wait --pipe --collect \
  --property=MemorySwapMax=0 --property=WorkingDirectory="$PWD" \
  build-exl3-venv/bin/python tools/glm_mixed_bench.py \
  --config configs/experimental/glm53f-q23-mixed.json \
  --model models/GLM-5.3-Flash-Q23-assembled.gguf \
  --output runs/mixed-q23
```

The benchmark requires the tokenizer Python dependencies. The example uses the
existing local virtual environment. Its idle check rejects other GPU compute
clients. On a desktop PC, `--allow-desktop-gpu` permits Chrome and desktop
contexts and records them; keep desktop activity consistent across controls.

The default protocol runs three independent processes for each configuration
and frozen 2K/8K prompt. Each process warms decode for 512 tokens, then times
512 tokens. Records include executable/source/prompt hashes, raw output,
timing, setup proportional set size (PSS), and cgroup memory. Mixed mappings are shared,
so summed RSS would double-count model pages. PSS is sampled during setup only
to keep page-table walks out of timing; cgroup memory and swap are sampled
throughout. Cgroup charges may omit file pages first charged elsewhere. Any observed swap rejects a
measurement. Use a cgroup with `MemorySwapMax=0` to enforce no swap.

The desired gains are 1.5x decode and 2x prefill relative to the faster fresh
single-GPU control using the same numerical profile. These are targets, not
measured results. Full paired corpus and HumanEval checks are required before
quality qualification. REAP224 is a secondary comparison, with its matching
prior. This runner remains experimental until those checks and repeated
measurements are recorded.

For existing corpus and HumanEval evaluation tools, the executable adapter is
`tools/glm_mixed_decode.py`. Select binaries with `STRATA_GLM_MIXED_PRIMARY`
and `STRATA_GLM_MIXED_WORKER`. Corpus quality configs using this adapter select
PSS in the memory guard. For HumanEval, pass `--prefill-experts=f16-batched` to
both the single and mixed runs, and use `glm_q2_run_guard.py --memory-metric=pss`
for the mixed run. Those checks retain the existing quality tolerances. The
adapter forwards the existing engine line protocol only for evaluation; it
does not change server configuration or installer support.


### Mixed runner optimization controls (TR16, 2026-10-14)

These switches are experimental and opt-in. `--combine strict` remains the
launcher default. `--combine device` adds the CPU and worker sums to the
primary GPU sum and changes floating-point addition order. Keep the same
choice in the single-GPU control. The evaluation adapter reads
`STRATA_GLM_CANON_COMBINE` so quality scoring uses the intended choice.

- `STRATA_GLM_MIXED_AFFINITY=1`: apply the existing routing affinity once to
  the union of the primary and worker residents. It changes routing. Ownership
  is refreshed before routing, while the expert service is idle. Both primary
  admission paths exclude worker-owned experts.
- `STRATA_GLM_MIXED_DECODE_GRAPHS=1`: capture worker products for repeated
  decode shapes, including repeated experts and verification batches. Keys
  include types, SwiGLU limit, gate skip, token count, group count and row count.
- `STRATA_GLM_MIXED_UPLOAD_THREADS=2|4`: use a bounded CPU copy pool for the
  worker's four pinned staging slots. GPU calls stay on its main thread. Zero,
  the default, performs the copies on that thread. Pending copies finish before
  their pinned buffers are freed.
- `STRATA_GLM_MIXED_PARTIAL_PREFILL=1`: for long FP16 prefill, accumulate
  weighted products on each GPU and return one FP32 vector per token. GPU
  atomics change addition order. The decoder applies the existing
  `STRATA_PREFILL_LOSSY_MIN_TOKENS` threshold, 4096 by default; short prompts
  keep route-row returns. Token partials have their own transport mode and
  destination validation. This mode has no quality qualification yet.

Mixed REAP-224 partitions use its actual 14 groups. The 18-group partition
remains the existing homogeneous runner's default. Host tests cover every
nonempty split for 14 and 18 groups. Transport tests cover sparse expert rows,
token partials, acknowledgements, stale replies and disconnects. CUDA and HIP
GPU tests cover weighted scatter with repeated token destinations, padding,
uneven expert lengths and an existing shared-expert sum.

Initial screens on TR16 used the Threadripper 1950X, user-reported DDR4-3200,
128 GiB RAM, RTX 5060 Ti 16 GiB and RX 9070 XT 16 GiB, with ROCm 10.1.
They used one fresh process per configuration, two 8192-token prefill passes,
a 512-token decode warmup and 512 timed tokens, with swap disabled. Desktop
GPU contexts and a background model transfer remained active. These are
preliminary timings, not quiet repeated speed gates or quality qualification.

| Model/profile | Placement | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Full Q23, strict, MTP1 | RTX 5060 Ti | 311.477 | 14.1324 |
| Full Q23, strict, MTP1 | RX 9070 XT | 454.1235 | 14.1147 |
| Full Q23, strict, MTP1 | RX primary, RTX worker | 192.6625 | 14.6853 |
| Full Q23, strict, MTP1 | RTX primary, RX worker | 203.0185 | 13.7735 |
| REAP-224, Q4 dense, MTP3, affinity 0.14 | RX alone, 1 GiB reserves | 607.504 | 30.8183 |
| Same lossy ingredients, union affinity, device combine | RX primary, RTX worker, 512-row panels | 244.243 | 25.878 |

The last mixed screen predates the fix to the second primary admission path
and worker graphs. It reduced timed CPU expert bytes from 732.87 to 508.53 GB
but was slower overall. Its cache ownership changes and worker overhead need
separate profiling. The CUDA lossy control stopped at the runtime-headroom
check before timing; a smaller staging ring is being evaluated.

An 8K diagnostic with row checks enabled passed after the admission fix,
batched worker graphs, four-thread staging and token partials were added.
It recorded zero swap. Across 42 layers, token partials returned 16 chunks
per layer instead of roughly 80–90; median output acknowledgement time was
27.27 ms instead of 98.84 ms. Diagnostic row checks add substantial CPU work,
so these phase timings are not prefill throughput measurements. A first unprofiled RX-primary/RTX-worker screen with these changes measured
335.8385 prefill tok/s and 26.7451 timed decode tok/s; it is one process and is
not quality-qualified. The RTX-primary/RX-worker screen subsequently measured
343.321 prefill and 32.0514 decode tok/s. The optimization targets are 50+ decode tok/s
and 1000+ prefill tok/s, with the user's 152/164 HumanEval floor; none is
established by these screens.


Further TR16 screens reused bounded pinned input buffers and moved next-layer
worker uploads after the current layer's final reply. The worker snapshots the
next layer before releasing the buffer lease and does not read that request
again while it preloads. `STRATA_GLM_MIXED_PREFETCH_NEXT=1` enables this for
prefill batches of at least 4096 tokens. `STRATA_GLM_MIXED_STABLE_OWNERSHIP=1`
keeps a disjoint worker set through primary promotions; overlap still triggers
an update, and prefill invalidates cached residency. Host tests cover both
cases and the reload after prefill. An 8K row-check diagnostic passed on both
vendor placements across these changes.

One-process 8K screens on the same hardware, with the earlier timing protocol:

| RTX primary / RX worker, REAP-224 Q23 and Q4 dense | Prefill tok/s | Timed decode tok/s |
|---|---:|---:|
| Unsplit MTP3, affinity 0.14 | 487.1185 | 32.1077 |
| Split MTP3, affinity 0.14 | 487.0565 | 35.1698 |
| Split MTP3, affinity 0.08 | 488.5975 | 34.9132 |
| Affinity 0.08, larger RX cache (33 experts/layer) | 498.7325 | 35.3913 |

These screens had no detected consecutive token cycle and zero recorded swap.
They have no corpus or HumanEval qualification yet. Unsplit verification did
not improve decode. The larger cache used a 14 GiB request, reduced to about
13930 MiB by observed free VRAM. Process-local CUDA pool trimming recovered
zero bytes and is not an established optimization.

A decode-only Nsight Systems attempt lost its agent after the first capture.
The raw trace was imported manually, but the second repetition is missing.
It is warmup diagnostic evidence, not a performance measurement. The 4096 x
4096 four-expert GEMM microbenchmark favored batched FP32 accumulation over
individual calls at 512 rows: 134.65 versus 109.29 TFLOP/s on RX 9070 XT;
47.21 versus 47.67 on RTX 5060 Ti. These are single-process synthetic GPU-event
means, not model throughput. The worker keeps batched products by default.
Paired corpus scoring completed on frozen binaries with 16 sequences and 8192
scored tokens. The full Q23 CUDA baseline measured perplexity 4.826016,
KL 0.136762 and top-1 agreement 0.877075. The REAP224 mixed candidate measured
5.020846, 0.191679 and 0.861450 respectively: perplexity increased 4.04%,
KL increased 0.054917, and agreement fell 0.015625. These exceed the standard
3% / 0.03 / 0.01 tolerances. The tool also rejects the pair as incomparable
model inputs because pruning changed the weights; this comparison includes
pruning and dense quantization, not just mixed execution. Both evaluations
completed, with zero recorded swap. Background CPU work was recorded, so these
are quality scores rather than timing evidence. The candidate is not qualified;
the 152/164 HumanEval floor remains required.

A subsequent same-weight control completed the same 16 sequences / 8192
targets on REAP224 with Q4 dense, single CUDA, strict reduction and no routing
bias. It measured perplexity 4.917685, KL 0.173008 and top-1 agreement
0.869141. Against this control, the frozen mixed affinity-0.08/device-combine
candidate increased perplexity 2.10%, KL 0.018671 and reduced agreement
0.007690. This pair passes the unchanged standard corpus tolerances, including
input and protocol checks. It isolates the mixed profile from pruning, but
does not qualify HumanEval or later prompt-aware cache and precision profiles.

Two further opt-in changes are under validation. With device combine,
`STRATA_GLM_MIXED_DECODE_PARTIAL=1` makes the worker reduce its selected decode
rows on the GPU and return one weighted vector per token. Strict combine keeps
the individual rows. The reduction changes addition order; its speed and coding
quality are not established. Row diagnostics also compare the returned sum
against a CPU sum of the same GPU rows.
Both CUDA and HIP GPU parity suites passed. A 128-token-prompt, 16-token MTP3
CUDA-primary/HIP-worker diagnostic completed with row and weighted-sum checks
enabled and zero recorded swap. This is functional evidence; its diagnostic
overhead makes it unsuitable as a speed measurement. One fresh 8K run with
512 warmup and 512 timed decode tokens measured 502.463 prefill tok/s and
35.6371 decode tok/s. Its timed CPU expert reads were 610.50 GB and layer flow
was 8986.3 ms. There was no detected token cycle or recorded swap. The generated
continuation differs from the earlier partial-row run, so the small timing
difference does not establish a speedup. Corpus and HumanEval qualification of
this generation remain pending.

`STRATA_GLM_MIXED_ASYNC_PANELS=1` removes the worker's wait after each long
partial-prefill panel. It still waits at each expert group before reusing pinned
metadata. Row diagnostics retain their panel waits. This change has not yet
been quality-qualified. Both CUDA and HIP worker builds passed. One fresh 8K
run measured 488.6345 prefill tok/s and 37.7171 decode tok/s with 512 warmup
and 512 timed decode tokens, no detected cycle and zero recorded swap. This
single run did not improve prefill over the synchronous-panel screen. Its
continuation changed, so the higher decode timing is not an established gain
from the prefill change. Both switches default to off.

`STRATA_GLM_MIXED_DEQUANT_ONCE_PRIMARY=1` is a further unmeasured prefill
experiment. For long mixed partial prefill it keeps separate FP16 gate/up and
down weight panels on the primary GPU, reusing them across token tiles. This
avoids repeated weight dequantization while preserving the products and
scatter operations. The extra down panel is released with the compacted
prefill workspace. CUDA/HIP builds and host/transport tests passed. One fresh
8K screen measured 503.325 prefill tok/s and 37.1102 decode tok/s, with 512
warmup and 512 timed decode tokens and no detected cycle. This does not
establish a gain. Across the whole process, client waits were 1117.58 ms for
decode replies, 1052.32 ms for ownership requests and 13286.8 ms for prefill
chunks; these include warmup and setup, not just the timed interval. Decode
partial returns totaled 790.66 MB. The switch defaults off.

`STRATA_GLM_MIXED_PROMPT_PRIOR=1` is an unmeasured cache-selection experiment.
The primary sends its full per-layer prompt route counts with an ownership
request, using the otherwise unused route-weight entries. A shared C11 helper
validates those nonnegative finite counts. The worker blends them with a
256-token prior, ranks complementary experts and keeps its existing per-layer
slot budget. Stable ownership still avoids repeated reloads during decode.
This can change routing when union affinity is enabled, so it needs fresh
quality checks. The switch defaults off. CUDA/HIP builds and shared host/transport
tests passed. One fresh 8K full-prompt screen measured 504.4845 prefill tok/s
and 36.9346 decode tok/s with 512 warmup / 512 timed tokens, 545.10 GB of timed
CPU expert reads and 8003.14 ms layer flow. No cycle was detected. This does
not establish an improvement or coding-quality qualification.
The transport test checks the prompt-count payload across the ownership update
and prefill invalidation. A second queued screen enables the existing
`STRATA_GLM_DECODE_CACHE_RECENT=1`, using the last 256 prompt tokens for both
primary and worker selection instead of the full prompt. Neither selection
experiment has a quality qualification yet. The recent-window screen completed
one fresh 8K run at 504.5335 prefill tok/s and 37.4586 decode tok/s, with 512
warmup / 512 timed tokens, 579.08 GB timed CPU expert reads and 8529.79 ms layer
flow. No cycle was detected. It did not exceed the earlier decode screen;
neither selection experiment establishes a gain.

The first partition sweep used one fresh process per placement, two 8K prefill
passes, and 512 warmup plus 512 timed decode tokens. All used the same REAP224
Q23/Q4-dense/MTP3 profile with partial decode and asynchronous worker panels:

| Primary / worker | Expert groups | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| RTX 5060 Ti / RX 9070 XT | 4 / 10 | 474.816 | 36.3506 |
| RTX 5060 Ti / RX 9070 XT | 5 / 9 | 485.097 | 34.8828 |
| RX 9070 XT / RTX 5060 Ti | 11 / 3 | 592.6535 | 34.3471 |

These are unqualified single-process screens. AMD-primary is the fastest mixed
prefill placement measured here, but it does not exceed the earlier AMD single
screen's 607.504 tok/s. Expert splitting has not established a prefill gain.

Reducing the primary's additional tier runtime reserve from 1024 to 768 MiB
admitted 654 slots / 5681.62 MiB instead of 629 / 5464.44 MiB. One fresh 8K
screen measured 500.489 prefill tok/s and 37.4744 decode tok/s, with 512 warmup
and 512 timed tokens. Timed CPU expert reads were 562.98 GB. Sampled free NVIDIA
memory stayed near 1.7 GiB during decode; there was no detected cycle or recorded
swap. Physical and allocation limits remained enforced. This single screen did
not exceed the earlier 37.7171 decode screen and is not quality-qualified.

A single cold 8K CUDA-primary prefill Nsight capture completed and exported
kernel, API and memory-transfer summaries. Instrumented prefill took 22.076 s;
this is diagnostic evidence rather than qualified throughput. CUDA kernel time
totaled about 12.1 s: the largest FP16 GEMM family took 4.118 s (34.1%), KDA
rows 1.085 s (9%), and flat dequantization 0.589 s (4.9%). Host-to-device copies
totaled 49.43 GB / 6.854 s, with overlap between copy and compute streams.
These durations must not be added as a wall-time estimate. Dequantization
reuse alone cannot explain the full prefill gap. The existing CUDA-only
`STRATA_PREFILL_F16_ACCUM=1` and `STRATA_PREFILL_F16_ACCUM_BATCHED=1` options
are queued for a long-prefill screen; they are lossy and need a precision
comparison before qualification.
That screen completed one fresh process at 526.2425 prefill tok/s and 37.1647
decode tok/s with 512 warmup / 512 timed tokens and no detected cycle. This
does not reach the prefill target or establish a repeated gain. Its paired
8K precision evaluation failed its fidelity gate; HumanEval remains unverified.
A paired precision evaluation completed for the frozen 8K code prompt plus
an EOS target, scoring 8192 positions with a single 8192-token prefill batch.
It saves control logits and compares FP16 accumulation directly against that
control, with routing bias and decode caching disabled in both runs. The
existing 3% perplexity / 0.03 KL / 0.01 top-1 tolerances are retained. An
incomplete or failed precision result excludes that profile from the queued
HumanEval selection; a pass still needs the 152/164 coding floor and repeated
throughput measurements.
The control measured perplexity 39.4440. FP16 accumulation measured 38.6289,
but direct KL was 0.405527 and top-1 agreement 0.841309, failing the KL and
agreement limits. Lower perplexity alone does not pass this comparison.

The existing MLA absorb/output/index FP16 chain was also screened with FP16
accumulation: 551.481 prefill tok/s and 35.3627 decode tok/s in one fresh 8K
process. Its comparison against the same saved control scored all 8192
positions at perplexity 38.7273, KL 0.385642 and agreement 0.844727. It also
failed. Both precision variants are excluded from HumanEval selection.

The first precision attempt stopped on incorrectly space-separated corpus
IDs; it was repeated with comma-separated IDs in a new directory. The first
HumanEval attempt stopped before scoring because the adapter did not accept
the server's `--serve` flag. The adapter now accepts `--serve` and the old
`--worker` alias, forwarding the runner's actual `--serve` mode. Four fake
engine launch cases passed, including failure cleanup and both aliases. The
failed evidence was retained. The corrected full HumanEval completed all 164
tasks on the frozen non-precision depth-three profile: 149/164 passed. This
misses the required 152/164 floor, so its 37.7171 decode tok/s screen does not
qualify the profile for use.

`STRATA_GLM_MIXED_MERGE_EARLY=1`, with strict combine and the existing
`STRATA_GLM_MOE_MERGE_GROUPS=early`, enables an opt-in experiment.
The worker returns individual canonical rows for both split-verification
groups before the CPU group reducer runs. Its complementary ownership prevents
those writes from overlapping primary-resident rows. The CPU shares expert
reads across the groups and can publish group A's sum before group B finishes.
CUDA/HIP builds and CPU protocol tests passed. The real-model row diagnostic
passed with 234 merged layers and no swap. One paired 8K screen measured
503.314 prefill / 35.6263 decode tok/s for the strict control, and 500.9745
prefill / 26.6911 decode tok/s with early merge. Both used 512 warmup and 512
timed decode tokens. Early merge was slower in this screen; it has no corpus
or HumanEval qualification.

Increasing MTP depth from three to four did not help the first 8K screen:
493.6315 prefill tok/s and 34.6826 decode tok/s, with 512 warmup and 512 timed
decode tokens. The timed run produced 512 tokens in 135 rounds, accepting
376 of 540 drafts, but read 626.92 GB of CPU expert weights. This is one
unqualified run. Depth five then measured 496.667 prefill tok/s and 33.0347
decode tok/s: 512 timed tokens in 124 rounds, 387 accepted of 614 proposed,
679.51 GB of CPU expert reads and 10061 ms layer flow. It also did not improve
decode. Depth seven measured 503.15 prefill tok/s and 26.4872 decode tok/s:
512 timed tokens in 123 rounds, 388 accepted of 852 proposed, and 866.14 GB
of CPU expert reads. It was slower again. All three longer-window screens
completed without a detected cycle; none establishes a quality-qualified gain.
Unsplit depth five then measured 499.7505 prefill tok/s and 27.7431 decode
tok/s in one fresh process with the same 512 warmup / 512 timed protocol.
It produced 512 timed tokens in 119 rounds, accepting 391 of 592 drafts.
It did not exceed split depth five or the depth-three screen, so the queued
unsplit depth-seven trial was skipped. No token cycle was detected.


Fresh single-GPU controls with the frozen depth-three mixed-screen build and
its non-precision profile completed on TR16. Each case used a separate process,
512 warmup and 512 timed decode tokens, and two timed prefill passes. These are
single screens with desktop contexts, not repeated speed or quality claims.

| Prompt | GPU | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| 2K holdout-code | RTX5060Ti | 167.0585 | 30.0609 |
| 2K holdout-code | RX9070XT | 240.8635 | 29.6386 |
| 8K holdout-code | RTX5060Ti | 448.5700 | 30.5332 |
| 8K holdout-code | RX9070XT | 554.7405 | 29.3132 |

These controls use affinity 0.08 and do not reproduce the earlier single-GPU
best configuration. They therefore cannot establish absence of degradation
against that earlier 34 tok/s result. Raw records are under the local untracked
`build-glm-release/mixed-reap224-single-controls-2k8k-v1/` directory.


`STRATA_GLM_MIXED_ADAPT_CACHE=1` enables a further opt-in worker cache
experiment. The primary accumulates route counts with a 128-input-token
half-life and sends a refresh every 128 input tokens per canonical layer.
The worker blends those counts with 512 routes of calibration prior and
retains its best old owners, replacing at most four per layer unless primary
ownership displaced more. The cache budget and per-layer slot count stay
bounded. `STRATA_GLM_TIER_LEARN_UNBIASED=1` collects the unbiased router's
choices; this combination rejects merged verification groups. The default
leaves this feature off. Both backend builds and both host protocol tests
passed. The first 128-token diagnostic stopped on a layer-44 native IQ
row: CPU/float L1 error was 3.1190%, GPU/float error 1.2907%. The CPU portion
exceeded the unchanged 3% row gate, so this diagnostic failed before cache
refresh was verified. Its raw output is retained. A shorter diagnostic uses
`STRATA_GLM_MIXED_ADAPT_INTERVAL=16` to exercise refresh; the normal interval
stays 128 (accepted range 1–4096). Passing that shorter test does not resolve
the longer native-row failure. Throughput and full quality evaluation remain
pending.

A prior-only calculation found little benefit from uneven worker cache
allocation: across 42 GU projections, redistributing the same 33 slots per
layer improved ideal normalized hit probability by only 0.15–0.29 percentage
points. This assumes equal expert sizes and an ideal prior-ranked primary
cache with 0, 10 or 14 slots per layer; it excludes projections absent from
the prior. It is neither a runtime speed measurement nor a quality result.
The calculation is retained locally as
`build-glm-release/mixed-validation/global-cache-prior-estimate.json`.


Nonmixed token comparisons against the pre-change decoder completed on CUDA
and HIP: each backend produced identical 16-token output for one 128-token
prompt, with no `STRATA_` environment switches and no mixed worker. All four
runs exited successfully without sampled swap. This verifies that short
workload only; it is not a throughput comparison or a proof for all settings.
The frozen adaptive generation before the interval knob was tested. The second
comparison for the revised generation was stopped while waiting to prioritize
the next throughput experiment; that generation has not had this check. The local result is
`build-glm-release/mixed-validation/nonmixed-default-parity-v1/result.json`.


The revised ownership diagnostic passed with a 128-token prompt, 32 requested
decode tokens, row checks enabled and refresh interval 16. It completed 1019
worker requests, returning 4094 decode rows and 21094 prefill rows. Decode
row checks covered the returned rows; prefill checks sampled up to four rows
per layer. The worker
recorded 78 adaptive updates and 312 new owners, four per update in this
sample. The run exited successfully with zero sampled swap. This verifies
refresh and row arithmetic on this short workload; it does not resolve the
failed longer native CPU/float check. Throughput screens use the normal
128-token interval and remain unqualified pending full quality evaluation.


The first adaptive 8K screens used 512 warmup and 512 timed decode tokens,
MTP depth three and the normal 128-input-token refresh interval. Each is one
process; none is quality-qualified.

| Profile | Prefill tok/s | Decode tok/s | Timed CPU expert GB |
| --- | ---: | ---: | ---: |
| Affinity 0.025, top-4 acceptance, adaptive off | 498.3590 | 23.9729 | 851.32 |
| Same settings, adaptive on | 498.9565 | 23.3256 | 739.50 |
| Adaptive, affinity 0.08, unbiased learning | 499.4645 | 26.1134 | 561.72 |

The low-affinity adaptive screen reduced CPU reads but added worker waits:
whole-process ownership wait grew from 1059.56 to 4314.64 ms, and decode wait
from 715.95 to 2450.73 ms. Those counters include warmup and differ from the
timed CPU-read counter. The fastest adaptive screen's HumanEval was deliberately
stopped after 13 passing tasks to prioritize a faster implementation; it is
incomplete and does not meet the 152/164 gate.

`STRATA_GLM_MIXED_ADAPT_AFTER_ROWS=1`, with adaptive caching enabled, moves
cache replacement after the current expert rows. The refresh control retains
current ownership; the worker computes and checks those rows, then uploads the
next owners and publishes ownership while it still holds the response lease.
Uploads and later expert kernels remain ordered on the same GPU stream.
This lets uploads overlap the primary's remaining current-layer work. The
flag defaults off. CUDA and HIP builds passed; row diagnostics and performance
measurement are pending. No speedup is established yet.


The first after-row diagnostic also stopped on a native layer-44 CPU/float
comparison: CPU L1 error 3.0004%, GPU L1 error 1.3584%. That failed record is
retained. `STRATA_GLM_MIXED_CPU_REFERENCE=ggml` adds a separate diagnostic
scope: CPU GU values are checked against independent ggml quantized dots,
then the down stage is checked using the CPU path's actual quantized hidden
values. Both stage comparisons require L1 relative error below 1e-5. This
avoids treating an activation-quantization difference against full float as
an implementation regression. CPU/full-float error is still reported; the
GPU/full-float 3% limit and canonical bitwise checks are unchanged. The new
scope does not retroactively pass either failed float-reference diagnostic.
Full corpus and HumanEval gates remain unchanged.


The separate quantized-arithmetic diagnostic passed on the short 128-token
prompt / 32-token decode workload, refresh interval 16. It returned 3978
decode rows and 21094 prefill rows, with 78 cache commits and 312 replacements.
Native CPU GU/down comparisons against ggml had maximum relative error
2.21211e-7; reported CPU/full-float error reached 3.29326%. Those maxima cover
their respective diagnostic scopes. Canonical bitwise row checks and the
GPU/full-float gate passed. Prefill checked up to four rows per layer. This
is not full-model quality qualification. The after-row timing screens and
complete corpus / HumanEval checks remain pending.


Further after-row screens measured 503.2995 prefill / 26.4948 decode tok/s
with affinity 0.08 and unbiased learning, and 513.9420 / 29.7036 with affinity
0.14 and top-6 MTP acceptance. The latter read 369.94 GB of CPU experts in the
timed decode, but whole-process decode wait was 5794.58 ms and ownership wait
3409.37 ms. These remain single, unqualified screens.

`STRATA_GLM_MIXED_UPLOAD_STREAM=1` enables a separate worker upload stream.
Each cached entry carries a readiness event; its first execution queues a
wait on the execution stream. Later uses retain that ordering. Pinned staging
slots wait for their copy event before reuse, and the copy stream is drained
before staging destruction. `STRATA_GLM_MIXED_REUSE_ADAPT_SLOTS=1` recycles
same-size allocations removed during after-row replacement, avoiding repeated
allocation and free calls. Both flags default off. Both backend builds passed;
GPU row diagnostics and controlled isolated/combined measurements are queued.
The current full HumanEval continues on its frozen earlier generation. A cold
single-AMD prefill trace is also queued to locate remaining prefill costs;
instrumented timings will not be used as performance claims.


During the long after-row HumanEval, worker logs showed roughly 11.5 GiB of
cached weights versus about 13.9 GiB near startup. The cause is not yet
established. To test graph-memory pressure,
`STRATA_GLM_MIXED_FIXED_GRAPH_GEOMETRY=1` captures worker kernels with capacity
`8 * nt` for groups and rows, while actual counts remain in metadata. The
kernels guard against those actual counts. This reuses graph variants across
route-count combinations; it does not change expert arithmetic. The flag
defaults off and only applies to the worker's batched graph path. Both CUDA
and HIP decoder/worker builds passed. Row diagnostics and measurements are
queued, and worker summaries now record graph counts. The four copy/reuse
screens hold fixed geometry enabled throughout; comparisons with the earlier
screens therefore cover the combined changes.


The full after-row aggressive-profile HumanEval completed all 164 tasks with
152 passes, meeting the user's 152/164 coding floor. Its earlier single 8K
screen was 513.942 prefill and 29.7036 decode tok/s, so it still misses both
throughput targets. This qualifies that coding result only; the full corpus
check for this generation was deferred to prioritize the next scheduling
experiment. The scoring run records CPU interference and makes no timing
claim. Its local evidence is
`build-glm-release/mixed-validation/humaneval-after-rows-selected-v2/measurement.json`.


The combined copy-stream/reuse/fixed-geometry diagnostic passed on the short
workload. It returned 6624 decode rows and 21094 prefill rows, with 78 updates,
312 replacements and six worker graph variants. Maximum native CPU/ggml stage
error was 1.81811e-7; GPU/float and canonical checks passed. Prefill checks
sample up to four rows per layer. Runtime timings remain pending.

The instrumented single-AMD cold trace recorded 9.44398 seconds of kernels,
6.55341 seconds of copies and 1.48045 seconds of overlap across the whole
process, including initialization and one decode token. Kernel families are
retained in its local summary; the largest GEMM family took 1.03447 seconds,
KDA row kernels 0.64096 seconds and flat dequantization 0.58918 seconds.
These totals are not qualified throughput. The final lower-affinity copy
screen also enables the existing `STRATA_GLM_PREFETCH_STREAM=1`; its comparisons
with the aggressive profiles include that additional prefill scheduling change.


The first fixed-geometry screens measured 509.2345 prefill / 35.2414 decode
tok/s with a separate upload stream, and 510.2960 / 35.2639 with allocation
reuse on the original execution stream. Timed CPU expert reads were 353.79
and 340.79 GB respectively. Both recorded six worker graph variants. Each
is one independent process with 512 warmup and 512 timed tokens; different
continuations and combined changes prevent attributing these differences to
one feature. Neither has new full quality qualification or reaches the targets.


The combined fixed-geometry/upload-stream/reuse screen measured 513.2505
prefill / 35.0263 decode tok/s. The final lower-affinity top-4 screen, also
with primary streaming prefetch, measured 509.4215 / 30.4776. These are single
screens and do not establish a prefill improvement. The fastest new screen
was allocation reuse at 35.2639 decode tok/s; its full coding and corpus checks
are now queued/running. None reaches 50 decode or 1000 prefill tok/s, and none
is promoted on throughput evidence alone.


The requested eight-hour optimization window ended without reaching 50+
decode or 1000+ prefill tok/s. The latest selected screen is 35.2639 decode /
510.296 prefill tok/s, with full validation still running. An earlier generation
met the 152/164 coding floor at 29.7036 decode tok/s. No new default is selected,
no throughput winner is promoted, and repeated timing qualification remains
incomplete. Already queued validation continues; additional optimization time
requires the user's decision.


The selected fixed-geometry/reuse profile completed all 164 HumanEval tasks
with 149 passes, below the required 152. Its 35.2639 decode / 510.296 prefill
screen therefore remains unqualified. The separate earlier after-row generation
scored 152/164 at 29.7036 decode tok/s. Different cache scheduling and
continuations mean this comparison does not isolate one change's effect.
The full corpus check of the latest selection is still running. The raw coding
record is retained locally under
`build-glm-release/mixed-validation/humaneval-copy-fixed-selected-v1/`.

The selected profile's corpus evaluation subsequently completed all 8192
positions. It failed the unchanged same-weight limits: perplexity 5.097743
(limit 5.065215), KL 0.210995 (limit 0.203008), and top-1 agreement 0.849487
(minimum 0.859141). Peak process-tree PSS was 87812332 KiB, with no swap or
memory-guard rejection. The command returned 1 for the quality failures.
Raw evidence is in
`build-glm-release/mixed-validation/quality-copy-fixed-selected-v1/candidate/result.json`.

### Simulator-first redesign, October 14

The upstream merge, measured resource model and revised design are documented in
[GLM_TR16_MIXED_REDESIGN.md](GLM_TR16_MIXED_REDESIGN.md). Its bounds distinguish
fixed-route transport limits from conditional kernel speedups and measured runs.

The user authorized a new twelve-hour redesign window, with performance
screening before full quality checks. The final coding floor stays 152/164.
`tools/glm_mixed_sim.py` provides an initial latency-budget and sensitivity
simulation using the recorded mixed MTP phases and fresh single-device 8K
screens. It preserves source hashes and explicitly labels hypothetical gains.
The original general simulator's GPU-count multiplier is insufficient for this
heterogeneous worker. The later resource/DAG path in `tools/sim/heterogeneous.py`
models explicit supplied scheduling, host transfers and shared bandwidth; use
the measured TR16 profile described in the redesign document above.

For the fixed-geometry/reuse screen, 512 decoded tokens took 14490.7 ms;
verification accounted for 12648.5 ms. Reaching 50 tok/s requires saving
4250.7 ms. If only verification improves, it must become 1.5062 times faster.
CPU layer-flow time is nested inside verification; the simulator sweeps its
unknown exposed share instead of adding that time to verification again.

The single-device prefill rates sum to 1003.3105 tok/s. A hypothetical fully
parallel work split assigns 44.709% to CUDA and 55.291% to HIP. At those rates,
only 27.03 ms of coordination fits inside the 8.192-second target for 8192
tokens. A conditional scenario with 1.25 times faster kernels, 10% serial work
and 500 ms coordination yields 1083.58 tok/s. This is a requirement for a
possible redesign, not a measured or validated forecast. It assumes the
whole prefill pipeline can be partitioned; current expert-only sharding does
not establish that assumption. Layer dependencies, live state, transport,
VRAM and shared host bandwidth still need stage-level measurements.

Reproduce the initial sweep with:

```bash
python3 tools/glm_mixed_sim.py \
  --mixed build-glm-release/mixed-reap224-copy-fixed-reuse-8k/results.json \
  --singles build-glm-release/mixed-reap224-single-controls-2k8k-v1/results.json \
  --output build-glm-release/mixed-validation/redesign-sim-v1.json
```

Calibration identity, balanced work partition, coordination overhead and
CPU-overlap boundary checks passed. Neither throughput target is achieved
by these simulator results, and no quality result is inferred from them.

The simulator also has a per-layer join model and a reader for synchronized
`STRATA_GLM_TASK_TRACE` records. The join model keeps each mixer and the first
three dense FFNs on the primary, then waits for both expert partitions before
advancing a layer. Scaling complete FFN times by expert share remains an
optimistic assumption because shared FFN work and staging do not scale that
way. Synthetic checks cover serial dense layers, parallel expert joins,
communication cost and mixer scaling.

The first redesign decode diagnostic stopped before inference: a 512 MiB
allocation with 1359.375 MiB free would have crossed the 1024 MiB runtime
reserve. Its raw failure is retained in `mixed-redesign-trace-v1` under the
local build directory. The queued prefill diagnostic was stopped to change
the same buffer configuration. Version 2 uses four CUDA prefetch groups
instead of six, retaining the runtime reserve. Decode slot tracing and
subsequent synchronized CUDA/HIP/mixed prefill traces are instrumented
diagnostics, not qualified throughput measurements.

The corrected decode trace completed with exit 0. In its timed pass, traced
host steps accounted for 13351.636 ms: host service 8875.2 ms (66.47%),
between-slot waits 3333.56 ms (24.97%), head 549.111 ms and tail 593.765 ms.
Host service includes worker ownership and completion waits, so it must not
be interpreted as CPU arithmetic alone. Large exposed waits appeared at MLA
slots; layer 3 also includes the initial dense-layer dependency. The full
instrumented decode was 32.3394 tok/s, with four CUDA prefetch groups. This
does not establish a regression against the six-group uninstrumented screen.
`tools/glm_mixed_trace_report.py` retains the scopes and ranks per-layer slots.
The raw run and report are in `mixed-redesign-trace-v2` and
`mixed-validation/redesign-decode-trace-v2.json` under the local build tree.

`tools/glm_mixed_prefill_sim.py` pairs synchronized layer traces with measured
prefill trials, discards the engine's extra warmup when `--bench` exceeds one,
and checks batch sizes, layer inventories and wall times before sweeping
partitions. The first completed CUDA diagnostic's two timed traces spent
8005/8003 ms in mixers and 10893/10874 ms in FFNs. The 8192-token target budget
is 8192 ms. At those instrumented costs, keeping CUDA mixers serial leaves
essentially no room for FFNs or transfers: expert partition changes alone
cannot reach the target. Mixer cost includes host submission, synchronization
and trace I/O; the figures are not isolated kernel times. HIP and mixed traces
are needed before selecting a redesign.

The HIP timed traces subsequently measured 5942/5932 ms in mixers and
8130/8140 ms in FFNs. With those costs, the optimistic per-layer model's best
expert-only split is about 708 tok/s with HIP primary (eight of fourteen
expert groups), assuming 10 ms communication per MoE layer. This conditional
model still scales complete FFN cost linearly and needs the measured mixed
trace to assess its optimism.

The first decode experiment from the host-service model is
`STRATA_GLM_MIXED_FUSED_RETURN=1`. It places selected-row reduction and its
readback inside the worker's resident graph, removing the separate host wait
before reduction. The same reduction kernel and order are used. Partial
return buffers are allocated only when this flag is active, and the graph
key distinguishes reduction and ordinary row-output captures. The existing
row-check path can still read all rows and verify the reduced result.
This flag defaults off. CUDA/HIP decoder and worker builds passed; a short
GPU row/reduction diagnostic is running against frozen binaries.

The completed mixed prefill traces averaged 16814 ms. The uncorrected
half-and-half per-layer partition model predicted 14101.25 ms, leaving a
2712.75 ms residual (equivalent to 64.589 ms per MoE layer). This includes
non-scaling shared FFN work, transport, contention and model error, rather
than identifying transfer time. Carrying this fixed residual into the HIP
primary model gives only 954.04 tok/s even with twice-faster mixers and
experts. That extrapolation is unvalidated, but it makes clear why the initial
10 ms coordination assumption was optimistic. The redesign must address
the residual costs as well as GPU work. The calibrated sweep is retained in
`mixed-validation/redesign-prefill-sim-v2.json` under the local build tree.

Worker profiling of the two timed mixed prefill passes recorded
2802.572/2801.065 ms in panel products/copies and 1295.133/1327.500 ms waiting
for 672 return-chunk acknowledgements. Request preparation was
148.435/152.009 ms. Next-layer prefetch runs outside these request scopes,
and worker phases overlap primary work, so the numbers cannot be added to
primary timing or directly subtracted as exposed savings. The frozen raw
breakdown is `mixed-validation/redesign-prefill-worker-breakdown-v2.json`.
A matching fused-return-off/on decode screen is queued after its row
diagnostic, using the same binaries, four CUDA prefetch groups and numerical
profile; full coding quality is deferred until performance screening.

The fused-return diagnostic completed with primary/worker exit 0 and no
swap. It returned 7568 decode rows and 21094 prefill rows, with 78 ownership
updates, 312 replacements and six graph variants. Maximum native CPU/ggml
stage error was 2.18494e-7. Its selected-row reduction matched the existing
descending FMA reference bitwise. This covers a 128-token prompt and 32
generated tokens, with sampled prefill row checks, not full model quality.
Evidence is `mixed-validation/fused-return-short-diagnostic-v1.result.json`.

The prefill transport experiment `STRATA_GLM_MIXED_PIPELINED_RETURN=1` adds
a second owned pinned/device staging slot. The primary acknowledges a shared
chunk after copying its payload and destinations into owned staging; the
worker can then fill the shared buffer while the primary's transfer stream
continues. Each slot has a completion event and is waited before reuse, and
the stream is drained before the layer returns. FP32 payloads, destination
validation and chunk order remain the same. The flag defaults off and adds
about 8 MiB each of device and pinned row staging when enabled. Builds are
queued after the fused-return timing pair to avoid compiler interference;
GPU validation and measured speedup are pending.

An architecture screen also modeled two contiguous layer stages with prompt
microbatches. It used the earlier matched frozen single-device 2K/8K rates,
scaled each measured batch cost by layer share, and assumed free boundary
transfers and ideal stage overlap. Four 2K microbatches modeled at 326.09
tok/s with 41% of layers on NVIDIA. One 8K batch approached only 553.43 tok/s
when almost all work stayed on AMD. Repeated weight streaming is already
included in the smaller-batch measurements. This model gives no reason to
implement that layer pipeline for the 1000 tok/s target; work sharing within
layers remains the candidate architecture. These are conditional simulations,
not measured multi-GPU results. The fill/drain formula is
`layer_pipeline_seconds` in `tools/glm_mixed_sim.py`, and the source-hashed
screen is `mixed-validation/redesign-layer-pipeline-screen-v1.json`.
