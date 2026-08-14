# LoD2 port - handover

Read `docs/lod2-port-spec.md` first: it is the contract.  `docs/lod2.md` is the
document for someone who wants to *use* LoD2 - flags, requirements, measured
results, code map.  This file is the engineering log: what exists, what is
verified, what was measured, what was tried and rejected, and what is next.

## 1. Where things are

| path | what |
|---|---|
| `/home/mose/Projects/KVM-paper/model/` | the author's implementation (do not edit) |
| `/home/mose/Projects/llm/LoD2-Qwen3.5-2B` | the wrapper that was actually evaluated |
| `docs/lod2-port-spec.md` | the algorithm and the port design |
| `docs/lod2-pytorch/lod2_ref.py` | standalone restatement (the executable spec) |
| `docs/lod2-pytorch/check_lod2_ref.py` | restatement vs author, and the refinement check |
| `docs/lod2-pytorch/dump_lod2_case.py` | writes oracle cases for the C++ test |
| `ggml/src/ggml-cpu/lod2.cpp` | CPU implementation of both ops |
| `src/llama-lod2.h` | the schedule, shared by the test and the graph builder |
| `tests/test-lod2.cpp` | replays an oracle case through the ops |

## 2. Verified

* restatement == author's `PagedTwoLevelLODAttention`: state bit identical,
  output 1.3e-6 at 1k and 4k (`check_lod2_ref.py`)
* refinement property: open every slot and LoD2 == dense, 5.7e-7
  (`check_lod2_ref.py --dense-check`)
* CPU ops == restatement on three cases including the paper's kernel schedule
  (`tests/test-lod2.cpp`), counts exact, outputs 2.1e-6
* the HIP kernels reproduce the same oracle cases as the CPU path, on all 8
  MI325X devices: output 1.25e-6, counts exact
* end to end: LoD2 with everything open produces greedy output identical to the
  dense path on Qwen3.5-2B (1500-token prompt, `-ub 512`), on both the CPU
  backend and `-ngl 99`.  With the default budget (routes 8/3, 16*sqrt(N)) the
  48 generated tokens also match dense on that prompt, which is a sanity check
  rather than evidence - the prompt is repetitive.

Reproduce:

```sh
cd docs/lod2-pytorch
python check_lod2_ref.py --len 1024 --decode 2 --device cpu --head-dim 64 --query-heads 4 --kv-heads 2
python check_lod2_ref.py --dense-check --len 1536 --device cpu --head-dim 64 --query-heads 4 --kv-heads 2
cd ../..
python docs/lod2-pytorch/dump_lod2_case.py --out /tmp/c.bin --len 1024 --decode 4 \
    --query-heads 4 --kv-heads 2 --head-dim 64 --device cpu
./build/bin/test-lod2 /tmp/c.bin
```

End to end (this is the strongest single check - it exercises the KV cache
storage, the graph schedule and the state recurrence against the real model):

```sh
head -c 6000 /home/mose/Projects/llm/lod-mk3-6k.txt > /tmp/p.txt
M=/home/mose/Projects/llm/Qwen3.5-2B-BF16.gguf
./build/bin/llama-cli -m $M -f /tmp/p.txt -n 32 --temp 0 -ngl 0 -c 4096 -ub 512 -st -no-cnv --simple-io
LLAMA_LOD2=1 LLAMA_LOD2_GROWTH=1000000 LLAMA_LOD2_ROUTES=100000 LLAMA_LOD2_ROUTES_PREFILL=100000 \
    ./build/bin/llama-cli -m $M -f /tmp/p.txt -n 32 --temp 0 -ngl 0 -c 4096 -ub 512 -st -no-cnv --simple-io
```

GPU work goes through `./simple-run.sh` (this shell has no `/dev/kfd` access).

## 3. Next step: performance (phase 2)

Phases 1a-1c are done.  What exists now:

* `LLAMA_LOD2=1` enables it; knobs are `LLAMA_LOD2_{ROUTES,ROUTES_PREFILL,
  LOCAL,CHUNK,PAGE,UPDATE,GROWTH,STATE_MIN,SINK,LAYERS,PAGES_PER_SLOT}`
* `prefill_chunk` is the ubatch, so `-ub 4096` reproduces the paper's window
* state lives in `llama_kv_cache::kv_layer` and is reset by `clear_lod2_state()`
* the graph is `build_attn(llm_graph_input_attn_lod2 *, ...)` in
  `src/llama-graph.cpp`; `can_reuse` is hard false for now

* `ggml/src/ggml-cuda/lod2.cu` implements both ops for HIP/CUDA (F16/F32 K/V).
  It borrows two things from the author's Triton kernels: base-two softmax
  (`exp2f`/`log2f` with a `log2(e)` factor folded into the scale) and the
  running (max, denominator, accumulator) formulation.  Unlike the Triton
  version the state update is **deterministic** - merges are accumulated by one
  block per slot in ascending column order rather than with atomics.

### Qwen3.6-27B Q4_K_M (arch `qwen35`, 65 layers, 16 full-attention, 24q/4kv, head_dim 256)

The port needed **no changes at all** for this model - same `qwen35` graph, same
kernels.  At 32k:

| check | result |
|---|---|
| refinement, all-open vs dense | **IDENTICAL** |
| mk3 needle, dense | 3/3 |
| mk3 needle, LoD2 16*sqrt(N) | **3/3** |
| mk3 needle, LoD2 32*sqrt(N) | **3/3** |

| 32k | prompt t/s | generation t/s |
|---|---|---|
| dense | 5676 | 75.5 |
| LoD2 | **783** | 41.2 |
| ratio | **0.14x** | 0.55x |

Generation holds the same ratio as on the 2B (about 0.55x), but **prefill fell
from 0.45x to 0.14x**.  The cause is structural, not a regression:
`lod2_k_attn` is one warp per (token, query head), and this model has 3x the
query heads and 2.7x the full-attention layers - 8x the units through a scalar
warp loop - while dense prefill is a tiled flash-attention kernel that scales
with head count.  What the 2B hid, the 27B shows plainly.

#### Shared-memory staging of the slot rows: tried, slower, reverted

The four warps of a prefill block are consecutive queries of the same head, so
they walk identical slots; staging each slot block in LDS should turn four
global reads into one.  It measured **worse on both models**:

| | before | staged |
|---|---|---|
| 2B prompt t/s | ~3910 | 3711 |
| 27B prompt t/s | 783 | 707 |

The premise was wrong.  The whole LoD2 state is about 27 MB per layer on the
27B and the device has 256 MB of infinity cache, so those "redundant" reads
were already cache hits costing almost nothing - and the `__syncthreads()`
pairs needed to stage them are pure added latency in a kernel that is latency
bound.  Reverted.

That also retires the reasoning that led there: the author's Triton `BLOCK_M`
tile is not buying raw traffic on this hardware, because the traffic is already
absorbed by cache.  Whatever it buys - register blocking, matrix-unit use - is
not something a shared-memory stage reproduces.

**This reorders the remaining work: the prefill local branch matters more than
the decode GEMV.**  Same prescription either way - stop hand-writing the
attention loops.  For prefill specifically, the local branch is ordinary causal
attention over `[bswa_begin, query_end)` and could use `ggml_flash_attn_ext`
outright if its logsumexp were available for the three-branch merge.

### Measured, Qwen3.5-2B BF16, MI325X

| context | | prompt t/s | generation t/s |
|---|---|---|---|
| 1.5k | dense | 518 | 297 |
| 1.5k | LoD2 all-open | 457 | 157 |
| 32k | LoD2, split decode path | 4013 | 94 |
| 32k | LoD2, + parallel topk/reduce | 4033 | 108.5 |
| 32k | LoD2, + NSMUL=2 | 4036 | 111.2 |
| 32k | LoD2, + tree merge in topk | 4050 | 141.3 |
| 32k | LoD2, + full waves (decode only) | 3968 | 144.3 |
| 32k | LoD2, + split cap removed, NSMUL=1 | ~3910 | ~147 (143.2 / 146.9 / 150.5) |
| 32k | LoD2, + maintained slot means | 3818 | **151.2** |
| 32k | dense | 8960 | 270 |
| 32k | LoD2 (16*sqrt(N), routes 8/3) | 3980 | 93 |

The split path below took generation from 33 to 93 t/s.  Templating the
accumulators on elements-per-lane changed nothing measurable (93.6 vs 94.0), so
they were not the bottleneck - worth recording, because it is the kind of fix
that looks obviously right and is not.

Prefill at 32k is 2.2x slower than dense, and that is expected for now: dense
prefill is a tiled flash-attention kernel using the matrix units, while the
LoD2 local branch is a scalar per-query loop.  Matching it needs a tiled local
branch, not a bigger budget.

### Generation path (implemented)

Generation takes a different route through `ggml_cuda_lod2_attn`, chosen when
the ubatch is at most `LOD2_DEC_MAX_Q` (8) tokens wide.  One warp per
(token, query head) is right for prefill and useless for a single token, so:

```
lod2_k_logits   one warp per (unit, slot)   -> logits[unit][s]      reads slot keys once
lod2_k_topk     one warp per unit           -> top_slots[unit][R]   reads logits only
lod2_k_part     one warp per (unit, split)  -> partial (m,d,acc)x3  reads slot values once
lod2_k_reduce   one warp per unit           -> dst
```

`unit = token*n_head_q + head`.  The column space
`[slots | local window | routed entries]` is cut into `NS` splits, and an
online softmax over a partition merges exactly, so the cut is free.  `NS` is
picked so `units*NS` is about four warps per CU, capped at 64 and at
`s_len/WARP_SIZE`; with one token and 8 heads on 304 CUs that is 512 warps
instead of 8.

Two consequences worth keeping:

* the slot keys are read **once** (in `lod2_k_logits`) and the slot values
  **once** (in `lod2_k_part`).  The prefill kernel still makes two passes,
  because storing the logits there would cost `n_tokens*n_head_q*s_len` floats.
* `lod2_k_part` and `lod2_k_reduce` are templated on the value elements per
  lane, so the three accumulators stay in registers.

The routed entries all live in split 0 rather than being cut: at 32k a slot
holds about 11 tokens, so a route is one page plus a residual, and a second
reduction would cost more than the work.

### Where the generation time actually goes (measured, not guessed)

Knob sweep at 32k, generation t/s.  Each knob changes exactly one part of the
read, so the deltas attribute the cost:

| configuration | prompt t/s | gen t/s | what it isolates |
|---|---|---|---|
| dense | 8803 | 264.8 | baseline |
| LoD2 routes 8/3, growth 16 | 3953 | 91.6 | default |
| LoD2 routes 1/1, growth 16 | 3953 | 91.3 | routed branch: 8 routes vs 1 |
| LoD2 routes 0/0, growth 16 | 4032 | 94.0 | routed branch off entirely |
| LoD2 routes 8/3, growth 4 | 5162 | 108.2 | 2896 slots -> 724 |

Reading the deltas:

* the **routed branch costs 2.5%** (91.6 -> 94.0).  The page scan was the
  obvious suspect and it is not the problem.
* cutting the slots **4x buys 18%** (91.6 -> 108.2), so the slot scan is about
  1.4 ms of a 10.9 ms token.
* dense is 3.78 ms/token, LoD2 is 10.9 ms.  Slots and routes together explain
  about 1.7 ms of the 7.1 ms gap.  **The remaining ~5 ms is not proportional to
  anything LoD2 reads**, so it is a fixed per-token cost.

Graph rebuilding is part of it, and it was measured rather than assumed:

| | gen t/s | ms/token |
|---|---|---|
| dense, graph reuse on | 270.1 | 3.70 |
| dense, graph reuse off | 217.8 | 4.59 |

So a rebuild costs about **0.9 ms/token**, and LoD2 was paying it on every
token because `can_reuse` was hard false.  That is now fixed (below), but it
only accounts for 0.9 of the 7.1 ms gap.

Budget as it stands, per generated token at 32k:

```
dense                       3.70 ms
LoD2                       10.90 ms
  slots (2896 of them)      1.87 ms   measured by the growth sweep
  routed branch             0.30 ms   measured by the routes sweep
  graph rebuild             0.90 ms   measured with LLAMA_GRAPH_REUSE_DISABLE
  local window (~512 cols)  ~0.3 ms   estimated from the slot cost per column
  UNACCOUNTED               ~3.8 ms
```

### The profile (rocprofv3, 32k prompt + 64 generated tokens)

```sh
./simple-run.sh bash /tmp/mose/lod2/prof.sh   # -> /tmp/mose/lod2/prof_out/lod2_results.db
```

Read it with sqlite; the tables are `rocpd_kernel_dispatch_<guid>` joined to
`rocpd_info_kernel_symbol_<guid>` on `kernel_id`.  Totals over the whole run:

| kernel | total ms | dispatches | ms per generated token |
|---|---|---|---|
| `lod2_k_attn` (prefill) | 3009 | 54 | - |
| `lod2_k_sim` (state update) | 292 | 528 | - |
| `lod2_k_merge` (state update) | 203 | 528 | - |
| `lod2_k_pages` (state update) | 173 | 570 | - |
| `lod2_k_part` | 157 | 384 | 2.45 |
| `lod2_k_topk` | 134 | 384 | **2.09** |
| `lod2_k_reduce` | 126 | 384 | **1.97** |
| `lod2_k_logits` | <51 | 384 | <0.8 |

384 = 64 tokens x 6 full-attention layers, so those four rows *are* generation.
`topk` and `reduce` together were 4.06 ms of a 10.9 ms token, and both ran on
**two blocks** of a 304-CU device: `topk` was one warp per unit doing R full
passes over `s_len` with a warp-argmax each time, and `reduce` was one warp per
unit walking `NS x 3 x Dv` floats serially.

This is the measurement that three rounds of reasoning failed to produce.  The
lesson worth keeping: the kernels that dominate were the two that do almost no
arithmetic, because they had almost no parallelism.

### What was changed in response

* `lod2_k_topk` is now one **block** per unit.  Every thread keeps a private
  top-R over its stride, then the block merges by walking the head of each
  thread's (already sorted) list, R rounds over `blockDim` heads instead of
  R passes over `s_len`.  The strides are disjoint, so no de-duplication is
  needed.
* `lod2_k_reduce` is now one **block** per unit with one thread per value
  dimension, so the splits are read coalesced.  This is why the split path now
  requires `Dv <= LOD2_BLOCK`; it falls back to the single-kernel path
  otherwise.

**Result: 32k generation went 94.0 -> 108.5 t/s** (dense is 269.7).  The
profile attributed 4.06 ms/token to those two kernels; the change recovered
about 1.7 ms of it, so both are still leaving time on the table:

* `lod2_k_topk`'s merge still runs on thread 0 alone (R rounds over `blockDim`
  heads).  Make it a tree reduction over the heads, or have each warp merge its
  own 64 lists first.
* `lod2_k_reduce` was rewritten again: merging partial softmaxes needs **no
  running chain**.  With `m = max_sp m_sp`, both the denominator and the
  accumulator are plain weighted sums with weight `exp2(m_sp - m)`, so it is a
  max pass (three threads, one per branch, through shared memory) followed by a
  dependency-free weighted sum.  Not yet measured - see below.

`lod2_k_part` (2.45 ms) is untouched and is now the largest generation kernel.
Its `NS` is capped at `s_len/WARP_SIZE`, which at 32k gives 45 splits = 90
blocks; that cap can be relaxed, and the GQA sharing below applies to it too.

### Correctness after the kernel work: confirmed

`/tmp/mose/lod2/verify.sh` strips the timing line properly (the earlier harness
did not, which is why it reported DIFFERS twice on runs whose generated text
was in fact identical - a harness bug reported as a regression).  With the
parallel `topk` and the rewritten `reduce` in place, on `-ngl 99`:

```
REFINEMENT all-open vs dense : IDENTICAL
default LoD2 vs dense        : IDENTICAL
```

So the split decode path, the block-per-unit `topk` and the chain-free `reduce`
all preserve the refinement property end to end.  Run this script after every
kernel change; it is the cheapest check that actually exercises the GPU path.

**A warning about benchmarking on this cluster**: two of these jobs landed on
the same node at once and dense measured 112 and 123 t/s in them, against 270
t/s when it had the node to itself.  Throughput numbers are only comparable
within a single job.

That accident is worth a second look, though.  Within those contended jobs the
ratio was:

| job | dense | LoD2 | LoD2/dense |
|---|---|---|---|
| alone on the node | 269.7 | 108.5 | 0.40 |
| contended | 123.5 | 104.0 | 0.84 |
| contended | 111.9 | 98.7 | 0.88 |

LoD2 barely moved (108.5 -> 104 -> 98.7) while dense lost more than half its
throughput.  That is what a read-volume advantage looks like under memory
pressure: dense streams 402 MB of KV per token at 32k and LoD2 streams a small
fraction of it, so when the memory system is shared dense is the one that
suffers.  Alone on the device, dense's traffic is served comfortably and LoD2's
kernel inefficiency is what shows.

The practical reading: the algorithmic win is real and already visible, and
what stands between it and beating dense on an idle device is the kernel work
listed below - not the amount LoD2 reads.

### GQA sharing in lod2_k_part: tried, measured, rejected

`lod2_k_part` was rewritten as one warp per (token, KV head, split) covering all
`G` query heads of a group, so a slot row is fetched once instead of `G` times.
On paper a 4x traffic cut on the largest generation kernel.  Measured at 32k:

| heads per warp | generation t/s |
|---|---|
| 1 (per query head) | 108.5 |
| 4 (whole GQA group) | **73.7** |

It is 32% *slower*.  Two reasons, both the same underlying fact - this kernel is
occupancy bound, not bandwidth bound:

* grouping divides the warp count by `G`.  Units went from `nq*Hq` to `nq*Hkv`,
  i.e. 8 to 2, and the split count cannot make that back once it saturates.
* `acc[G][3][EPL]` is 48 floats per lane at `G=4, EPL=4`, plus `G` copies of the
  q/logit/route pointers.  The register file spills.

This is the same lesson the slot sweep already gave (4x fewer slots bought only
18%): reducing what LoD2 *reads* is not what makes it faster right now.

The code keeps both: `lod2_k_part` is templated on `<G, EPL>` and `G` comes from
`LLAMA_LOD2_HPW`, defaulting to **1**.  The grouping becomes worth revisiting
only once the kernel is bandwidth bound - for example after the F16 mean cache,
or on a model with a much larger GQA group.

Split count is also tunable now (`LLAMA_LOD2_NSMUL`, default 1):

| config | generation t/s |
|---|---|
| HPW=1, NSMUL=1 | 109.1 |
| HPW=1, NSMUL=2 | 111.2 |

Refinement stayed IDENTICAL through all of it.  `NSMUL` now defaults to 2.

### State at the end of the session

32k generation is **154.6 t/s** against dense ~269.  Three runs of the identical
configuration gave 150.5, 146.9 and 143.2 - a +-2.5% spread.  **Treat anything
under about 5% as noise**, and note that several of the smaller steps in the
table above (144.3 vs 141.3, and 150.5 vs 149.7) are inside that band and should
not be read as improvements.  Only the big jumps are real: 33 -> 94 (split-K),
94 -> 141 (parallel topk/reduce plus the tree merge), and 141 -> ~147 (wave
width plus the split cap).  Four hypotheses have been
killed by measurement, which is what makes the remaining search tractable:

| tried | result |
|---|---|
| routed page scan | 2.5% of the token |
| register accumulators (templating) | no change |
| "it is all graph rebuild" | 0.9 ms of 7.1 |
| GQA sharing of the slot read | 32% *slower* |

### Second profile: reduce fixed, topk untouched

Same run, current kernels:

| kernel | before | after | verdict |
|---|---|---|---|
| `lod2_k_reduce` | 1.97 ms/token | **gone from the top 9** | fixed |
| `lod2_k_topk` | 2.09 ms/token | **2.15 ms/token** | **no change at all** |
| `lod2_k_part` | 2.45 ms/token | 2.17 ms/token | slightly better |

The `reduce` rewrite (max pass, then a dependency-free weighted sum) worked.
The `topk` rewrite did **not**, even though it went from one warp to a whole
block - 32x more threads for the same time.  So threads were never the
constraint there.

What is left, and it is the same shape as before: `lod2_k_topk` runs on
**`units` blocks = 8**, so it can occupy 8 CUs of 304 no matter how wide the
block is, and its wall time is one block's serial time.  Inside that block the
merge is still `thread 0` walking `R * blockDim` heads with no latency hiding
(one wave per CU).  359 us per dispatch is consistent with that.

Fixed by making the merge a **tree reduction** over the per-thread heads
(`R * log2(blockDim)` steps instead of `R * blockDim`), keeping the cursor trick
so each round advances only the winning thread's list.  The 8-block ceiling is
still there and did not need removing:

**32k generation 111.2 -> 141.3 t/s**, refinement still IDENTICAL.

Giving the kernel more blocks (grid `(units, NCH)` with a fourth merge kernel)
remains available if it shows up again in a profile.

### Third profile: only lod2_k_part is left

| kernel | per generated token |
|---|---|
| `lod2_k_part<1,8>` | **2.04 ms** |
| `lod2_k_topk` | gone from the top 8 |
| `lod2_k_reduce` | gone from the top 8 |
| `lod2_k_logits` | gone from the top 8 |

Generation is now one kernel plus small change.  At 32k the token is 7.08 ms
against dense 3.70; `lod2_k_part` is 2.04 of it and the graph rebuild another
0.9, so those two are the whole remaining gap.

`lod2_k_part` is 720 warps (8 units x 90 splits) each covering ~32 slots and ~6
local columns - about 46 MB per dispatch, which at HBM speed is ~9 us against a
measured 339 us.  So it is latency bound, not bandwidth bound, like everything
else in this kernel set.  Look at occupancy first: `acc[G][3][EPL]` is 24 floats
per lane at `G=1, EPL=8`, plus `val[8]`, `kk[8]`, `vv[8]`.  `EPL` is 8 because
`WARP_SIZE` is 32 in this build while the wave is 64, so each lane carries twice
the elements it would with a wave-width warp.

**Done and measured**: `lod2.cu` now uses its own `LOD2_LANES`
(64 under `GGML_USE_HIP`, `WARP_SIZE` elsewhere) for both the lane mapping and
the shuffle width.  On CDNA that turns every half-wave "warp" into a real wave,
which halves `EPL` from 8 to 4 and with it `acc[G][3][EPL]` and the `val`/`kk`/
`vv` staging.  The wave count is unchanged - two 32-lane groups were already
sharing one wave - so this is a pure register saving.  Decode went 141.3 -> 144.4 t/s, but prompt fell 4050 -> 3597: the *prefill*
kernel wants the opposite trade.  It has thousands of units, so two half-wave
groups per wave give it more units in flight, and that beats the register
saving.  So the lane width is now per kernel - `lod2_k_attn` keeps `WARP_SIZE`,
the decode kernels use `LOD2_LANES` - via a `lod2_wsum<L>` template.  Final:
prompt 3968, generation 144.3, refinement IDENTICAL.

Widening the wave exposed a bug in the split count: `NS` was capped at
`s_len/lanes`, so doubling the lane width silently **halved** the number of
splits (90 -> 45).  Lanes are spent on the head dimension, not on slots, so a
split owning fewer slots than a lane group still uses the whole wave; the cap is
now `s_len`.  That should let `nsmul*4*n_cu/units` govern again.  Change is in,
and the sweep picked the new optimum:

| NSMUL | gen t/s |
|---|---|
| **1** | **150.5** |
| 2 | 149.7 |
| 4 | 138.1 |
| 8 | 107.5 |

`NSMUL=1` now means `NS = 4*n_cu/units = 152` splits, three times what the cap
allowed.  Default changed back to 1.  Beyond that the `O(NS)` fold in
`lod2_k_reduce` plus the per-split fixed work costs more than the parallelism
buys - the same curve every knob here has.

The general shape, four times over now: **this kernel set is latency bound, and
anything that trades parallelism for locality loses.**  GQA sharing lost, wider
waves lost on prefill and won on decode, and the two big wins (split-K, tree
merge) were both pure parallelism.

Note also that `lod2_k_part` instantiated as `<1, 8>`, i.e. `EPL = Dv/WARP_SIZE
= 8`, which means `WARP_SIZE` is 32 in this build even though the hardware wave
is 64.  Results are correct (the shuffles are width-32 subgroups), but every
"warp" here is a half wave; worth accounting for when reasoning about
occupancy.

### The earlier note on this change, kept for context

Share the slot read across a GQA group.  Four query heads read the *same* slot
rows, and `lod2_k_logits` currently gives each its own warp, so every slot row
is fetched four times.  One warp per (token, kv head) that loads the row once
and computes `g` dot products cuts the measured 1.87 ms slot cost to about
0.47 ms.  The same applies to `lod2_k_part` for the slot values.

Storing an F16 mean beside the F32 sums halves it again; the F32 sums are then
only needed for the R opened slots, where the residual requires them exactly.
Atomics are acceptable here (the author's kernels use them) - the determinism
in the current state update was a bonus, not a requirement.

### Full decode breakdown at the current best config

| kernel | ms/token |
|---|---|
| `lod2_k_part` | 1.66 |
| `lod2_k_reduce` | 0.38 |
| `lod2_k_topk` | 0.27 |
| `lod2_k_logits` | 0.07 |
| **LoD2 attention total** | **2.38** |

The token is 6.8 ms and dense is 3.7 ms, so the other 1.1 ms of the gap is
outside these kernels - the graph rebuild (0.9 ms, measured) plus launch
overhead.  **To beat dense, the attention has to fit in about 0.4 ms**, six
times less than now.

### Why six times is plausible, and what it takes

Dense reads 402 MB of KV per token at 32k and does it in roughly 0.4 ms - about
1 TB/s effective.  `lod2_k_part` reads about 212 MB per token and takes 1.66 ms:
**128 GB/s, an eighth of dense's efficiency.**  LoD2 wins by 2x on bytes and
loses by 8x on how it moves them.  The headroom is real and it is all in the
kernel, not the algorithm.

The structural reason: **the coarse branch is a GEMV** - `q` against the 2896
slot mean keys - and so is the local branch.  This port computes them with a
hand-rolled per-slot loop while llama.cpp already ships tuned GEMV kernels
(`mul_mat_vec_f`).  Maintaining the slot means as an F16 matrix next to the F32
sums would let both branches become `ggml_mul_mat` against that matrix, which is
the real argument for the F16 mean cache - the halved traffic is the smaller
half of the benefit.

That reframes the remaining work: not "tune these kernels further" but "stop
hand-writing the two GEMVs".

### Maintained slot means (done) and the F16 question (open)

The slot means are now maintained in a separate table, `lod2_s_mn`
`[D+Dv, S_cap, H_kv]`, published by a kernel at the end of every state update
(`lod2_k_means`) and read by `lod2_k_logits` and the coarse branch of
`lod2_k_part`.  The F32 sums are still there and are read only for opened
routes, where the residual genuinely needs them.  This removes a per-element
divide from both hot loops and is the prerequisite for the GEMV move below.
Measured 151.2 t/s against a ~147 baseline whose spread was 143-150, so the
gain is at the top of the noise band rather than clearly above it - do not
count it as more than a couple of percent.

**F16 was tried for that table and reverted.**  Against the oracle:

| mean table | output max_abs | mean_abs |
|---|---|---|
| F32 | 1.07e-6 | 3.2e-8 |
| F16 | **1.12e-2** | 1.9e-6 (60x worse) |

The 6k case failed the 2e-3 threshold.  This is not obviously a bug - the author
stores page summaries at INT8 and leaves at INT4, so a low-precision coarse
layer is within the design - but the mean error rising 60x is a real precision
change, not one unlucky outlier, and adopting it needs end-to-end evidence
(refinement plus NIAH), not an argument.  Left at F32; the experiment is one
`GGML_TYPE_F32` -> `GGML_TYPE_F16` away in `llama-kv-cache.cpp`, the two ggml
asserts, and the load sites in `lod2.cu` / `lod2.cpp`.

Note the traffic win people expect from F16 is the *smaller* half of the prize;
the real one is below.

### THE ANSWER, from the author's kernels (read these before touching anything)

`kernels/lod_kernels.py:3047`, `route_logits_topk_coarse_attention`, takes
`route_logits` as an **argument**: a `[B, QH, Q, S]` tensor that is already
computed when the Triton kernel starts.  It is produced by
`triton_lod_attention.py:_state_route_logits`:

```python
mean_k = self._mean(state_k[..., :state_len, :], counts[..., :state_len, :])
return torch.matmul(q.detach(), self._repeat_kv(mean_k).transpose(-1, -2))
```

That is a plain batched **GEMM**.  The author hands the D-dimension contraction
- the expensive part - to rocBLAS, and the Triton kernel only does the top-k
selection and the coarse combine on top of the finished logits.

**This port computes those logits with a scalar warp loop.**  That is why
prefill is 0.14x of dense on a model with 24 query heads: the contraction runs
at scalar-ALU rate instead of on the matrix units.

The fix in llama.cpp terms:

1. `logits = ggml_mul_mat(mean_k_view, q)` - a view of `lod2_s_mn` with
   `ne0 = D`, `nb1 = (D+Dv)*4`, against the permuted q.  `ggml_mul_mat`
   broadcasts src0's `ne2` over src1's `ne2`, which is exactly GQA.
2. the kernel keeps top-k, the coarse combine (it still needs `log(count)`,
   the opened-slot exclusion and the LSE), the local window and the leaves.

The maintained mean table `lod2_s_mn` added earlier is the prerequisite for
step 1 - `ggml_mul_mat` needs the means as a matrix, not as sums to divide.

Note the intermediate is large: `Q * QH * S * 4` bytes, about 1.1 GB for the
27B at `-ub 4096`.  **This was dismissed once on that basis and the dismissal
was wrong** - the author materializes exactly this, and 1.1 GB transient on a
262 GB device is not a constraint.

Process note, recorded because it cost eight failed experiments: the kernel
*bodies* under `kernels/` were not read until the very end.  The orchestration
layer and the config attributes were, and they do not contain this - the
signature of one Triton launcher does.

### GEMM routing logits (done): right diagnosis, one third of the work

`ggml_mul_mat` now produces `q . mean_k` for every live slot, and both the
prefill kernel and the decode path consume it (`lod2_k_logits`, a warp per
slot, became `lod2_k_bias`, which only adds `log2(count)`).  The mean table is
copied to a compact `[D, slots, H_kv]` first so the GEMM sees a contiguous
src0 - one 13 MB copy per layer.

| | before | after |
|---|---|---|
| 2B prompt | ~3980 | 4232 (+6%) |
| 2B generation | ~147 | **154.6** |
| 27B prompt | 783 | 870 (+11%) |
| 27B generation | 41.2 | 39.7 (noise) |

Real, but not a change of scale - and the reason is worth stating plainly,
because it is the map for the rest of the work.  **There are three contractions
in this read and this replaced the smallest:**

1. **the local window** - up to 4864 columns per query, still a hand-written
   scalar loop.  On the 27B this dominates; the author hands it to aten's
   flash attention, which also returns the logsumexp the three-branch merge
   needs.
2. **the coarse value side** - `sum_s p_s * mean_v_s`, also a GEMM in principle,
   still hand-written.
3. the routing logits - done here.

So "stop hand-writing the contractions" was the right diagnosis and it is one
third executed.  Next is (1), and the obstacle is concrete: `ggml_flash_attn_ext`
does not return the logsumexp, so it needs either an optional LSE output added
upstream (llama.cpp's FA kernels already track `(max, denominator)` internally
and carry them through the split-K combine) or a tiled FA written here.

### The next change, and why the sweep points at it

`lod2_k_part` is 2.04 ms/token and latency bound.  The obvious response is more
splits, and the sweep says no: 150.5 / 149.7 / 138.1 / 107.5 for NSMUL
1 / 2 / 4 / 8.  That decline is monotone and steep, which is the signature of a
**serial term proportional to NS** somewhere downstream - and there is one.

`lod2_k_reduce` is one block per unit, one thread per value dimension, looping
over all `NS` splits.  Eight blocks, and each thread walks `NS * 3` partials in
order.  At NS=152 that is already 456 iterations on 8 CUs; doubling NS doubles
it, which is why NSMUL=2 and beyond lose more than the extra parallelism in
`lod2_k_part` gains.

So the two are coupled, and the unlock is to fold the splits in parallel:
a tree reduction over splits (grid `(units, 3)` or a two-stage fold) instead of
a per-thread serial walk.  Then `NS` can grow, and `lod2_k_part` - currently
1216 warps, about 4 per CU, which is thin for latency hiding - gets the
occupancy it wants.

Predicted, not measured.  Re-profile after changing it; the last three profiles
each moved the bottleneck somewhere new.

### Remaining levers, after that

1. **Store slot means in F16 next to the F32 sums.**  Attention reads
   `key_sum/count` and `value_sum/count` for every live slot every token; a
   maintained F16 mean halves that traffic and is what LoD1 already does with
   `k_mean`/`v_mean`.  Worth about the 18% the slot sweep showed, doubled.
2. **Share the slot read across a GQA group.**  Four query heads read the same
   slot rows; one warp per (token, kv head) computing four dots would cut the
   slot traffic 4x.
3. **Prefill.**  At 32k LoD2 prefill is 2.2x slower than dense (3953 vs 8803).
   The slot sweep moves it (5162 at growth 4), so part is the two-pass slot
   scan, but the rest is that the local branch is a scalar per-query loop while
   dense prefill is a tiled flash-attention kernel on the matrix units.
   Matching dense here needs a tiled local branch.

### Old notes on the integration, kept for context

Files to touch and the reason:

1. `src/llama-cparams.h`, `src/llama-context.cpp` - `LLAMA_LOD2` env plus the
   schedule knobs, per-layer enable.  Follow the LoD1 block at
   `llama-context.cpp:251`.
2. `src/llama-kv-cache.{h,cpp}` - allocate `s_kv`, `p_kv`, `p_idx`, `s_pg`,
   `meta` per layer next to `k_page`/`v_page` (`llama-kv-cache.cpp:261`), and
   keep `v_trans=false` as LoD1 already does.  Add the per-stream host scalars
   `lod2_coverage` and `lod2_slots`.
3. `src/llama-graph.{h,cpp}` - `llm_graph_input_attn_lod2` and a `build_attn`
   overload.  Per ubatch `[p0, p1)`:
   * `lb = params.local_begin(p0)`
   * plan and emit `ggml_lod2_update` nodes up to `lb` (should already be
     there from the previous ubatch, so normally zero nodes)
   * one `ggml_lod2_attn` with `l0 = lb`, `q0 = p0`, `s_len`, routes
   * plan and emit the updates up to `local_begin(p1)`
   * chain each update's result into the next op's `s_kv` src so the nodes are
     ordered by a real data dependency
4. `src/models/qwen35.cpp:331` already has the LoD branch point; add the LoD2
   one beside it.  Nothing Qwen specific may go into the graph code.

### The one design decision that needs care

`coverage` and `s_len` are **not** pure functions of the position: the update
block boundaries are clamped at each ubatch end, so they depend on how the
prompt was split.  Therefore:

* keep them as host state in the KV cache context, one per stream
* the graph **builder** reads them and bakes the plan into op params
* `set_input` advances them, so a reserved-but-never-computed graph
  (`graph_reserve` at startup) cannot corrupt the state
* `can_reuse` must return **false** for LoD2 graphs in phase 1 - the plan is
  baked in and differs per ubatch.  Revisit in phase 2.

### Phase 1 restrictions to assert, not ignore

one sequence, contiguous positions, f16/f32 cache, no context shift, no prefix
reuse, no speculative rollback (the state is a sum; rewinding needs an undo).

## 4. After that

* `ggml/src/ggml-cuda/lod2.cu` - both ops on HIP/CUDA.  `lod2_update` is a
  similarity matmul + an argsort of `M` values per head + a scatter-add;
  `lod2_attn` is one block per (token, query head).
* NIAH: compare against the reference numbers in
  `/home/mose/Projects/llm/LoD2-Qwen3.5-2B/docs/lod2-results.md`, using
  `-ub 4096` so the schedule matches the paper's exactly.
* INT4 leaf storage (`kv_bits=4`) is phase 2; it is a storage format, and the
  author measured it as lossless on 32k NIAH.

## 4b. Session of 2026-08-13: decode 39.7 -> 59.1 t/s on the 27B

All numbers Qwen3.6-27B-Q4_K_M at 32k, `-ub 4096`, model layer-split across
8 MI325X (dense measured the same way, so the comparison holds).  Every step
below kept `REFINEMENT all-open vs dense : IDENTICAL`, `default LoD2 vs dense :
IDENTICAL`, and the oracle at 1.07e-6 on CPU and ROCm.

| step | 27B gen | 2B gen |
|---|---|---|
| start of session | 39.7 | 154.6 |
| coarse branch: fixed max, no rescale chain | 53.4 | 173.8 |
| LDS fold of a block's splits | 54.2 | 178.3 |
| topk single-warp merge + branch-free reduce | 55.5 | 182.6 |
| topk absorbs prep (keep R+1) | 57.1 | 191.5 |
| per-KV-head logit kernel replaces GEMV+cont at decode | **59.1** | **195.4** |
| dense reference | 75.8 | 271 |

Prefill is unchanged (868 t/s vs dense 5570); nothing here touched
`lod2_k_attn`, which is still 74.8% of all GPU time during prefill.

### How to measure (do this before changing anything)

`rocprofv3 --kernel-trace --stats` writes a sqlite `.db`, not CSV.  Two runs at
`-n 1` and `-n 65`, diff the `top_kernels` view, divide by 64 -> ms/token of
decode.  **The `total_duration` column of that view is microseconds, not
nanoseconds** - taking it for ns says the GPU is 99.9% idle and sends you
chasing a launch-overhead problem that does not exist.  Scripts:
`/tmp/mose/lod2/dprof.sh` + the inline python in the session log.
`LLAMA_LOD2=0` still *enables* LoD2 (`getenv != nullptr`); unset it for a dense
baseline.  The dense `-n 65` run segfaults under rocprofv3; use `-n 17` with
`--kernel-trace` only (`/tmp/mose/lod2/dn.sh`).

### The decode budget, fully accounted

Dense: 6.74 ms/token GPU busy, of which flash attention is 0.97.  That 0.97 is
the number LoD2 has to beat, not the KV cache size - dense FA reads 134 MB per
layer at 2.2 TB/s.

At 59.1 t/s LoD2 is 16.9 ms/token wall, 9.38 GPU busy.  The 3.7 ms/token that
still separates it from dense:

| item | ms/token | note |
|---|---|---|
| graph rebuild (CPU) | ~1.0 | `can_reuse` still returns false |
| `lod2_k_part` over dense FA | 0.70 | 1.675 vs 0.97 |
| `lod2_k_sim` | 0.74 | state update, ~3 events per 64 tokens |
| `lod2_k_reduce` | 0.44 | |
| `lod2_k_topk` | 0.38 | |
| `lod2_k_logits_kv` | 0.37 | |

### What made the difference, and why

**The coarse branch was serialized on its own softmax.**  `acc = acc*co + pr*v`
with `co = exp2f(m_prev - m_new)` makes every slot depend on the previous one,
so the loop ran at the latency of a dependent transcendental chain - 2% of
achievable bandwidth however well the reads coalesced.  The logits are already
final before the value pass, so the branch maximum can be taken up front and
the accumulation becomes a plain weighted sum.  This one change was 4.1x on
`lod2_k_part` (8.44 -> 2.05 ms/token) and is the largest single win of the
session.  Two things fell out of it: opened slots are excluded by writing -inf
into the logit (exp2f(-inf - m) is already 0, so no per-slot route compare), and
empty slots need no count load, which removed a stride-2048 gather.

**The routed branch was pinned to split 0.**  A page scan plus a residual plus
16 exact columns per route, all in one warp, while the other splits walked ~76
slots each - and the kernel ends when the last warp does.  Route r now goes to
split r; logsumexp is associative so the result is unchanged.

**Split count and reduce cost were the same number.**  More splits meant more
parallelism in `lod2_k_part` but a proportionally longer serial fold in
`lod2_k_reduce`, which pinned NS to a value leaving 4 waves per CU.  A block now
owns LOD2_WPB consecutive splits of one unit and folds them in LDS, so the
reduce sees NS/4 partials.  Staging one branch at a time keeps LDS at
WPB*G*Dv floats instead of WPB*G*3*Dv.

**Data-dependent `continue` blocks pipelining.**  In `lod2_k_reduce`, skipping
empty partials made each iteration control-dependent on its own load.  An empty
partial has pm == -inf and exp2f(-inf - m) == 0, so weighting it in is free and
the loads pipeline: 0.845 -> 0.435 ms/token.  Same shape of fix in `lod2_k_topk`
(the insertion sort now sinks the candidate through constant indices).

**A block-wide tree merge costs a `__syncthreads` per level.**  R rounds came to
~64 barriers, and with one block per unit there is nothing to overlap them
with.  Warp-level merging costs shuffles and no barriers.  Keeping R+1 instead
of R made the old `lod2_k_prep` disappear entirely: the largest logit the coarse
branch keeps *is* the (R+1)-th largest, so the branch maximum comes out of the
merge instead of a second full pass.

**A GEMV is not a GEMM.**  Handing the routing logits to `ggml_mul_mat` is right
for prefill (mean table read once, reused across the tile: +11% there) and wrong
for decode, where one query turns it into a GEMV that reads the table once per
*query* head, plus a `ggml_cont` of the whole table - together 0.98 ms/token for
174 MB of traffic worth 0.09.  `logits` is now an optional src of
`ggml_lod2_attn`: null means "compute it yourself", and `lod2_k_logits_kv` reads
each mean row once per *KV* head.  0.98 -> 0.37 ms/token.  `n_tokens` is known
at graph build time, so each path gets the tool that suits it.

### Rejected by measurement this session (do not retry blind)

- **GQA sharing (`LLAMA_LOD2_HPW`)**: 59.0 (G=1) / 56.3 (G=2) / 54.0 (G=3) /
  46.0 (G=6), and worse again at higher NS.  Cutting mean-table traffic 6x makes
  it *slower*: `lod2_k_part` is warp-count bound, not bandwidth bound.  Note the
  old `{1,2,4,8}` ladder was unusable here anyway - a warp shares one KV head
  across its G query heads, so G must divide g_kv, which is 6 on this model.
  G=4 with g_kv=6 straddled two KV heads.  The dispatch now clamps G.
- **Split count**: flat from NS=38 to 152 (54.5 / 55.2 / 54.5), falling after
  (53.2 / 50.2 / 44.5 at 304 / 608 / 1216).  Default is 4 blocks per CU.
- **Growth factor**: 8 -> 59.1, 16 -> 57.0, 32 -> 52.1 at the time of the test.
  Halving the slot count buys only ~4%, so the coarse scan is no longer what
  decode is made of.
- **Scratch spilling**: checked directly via `private_segment_size` in the
  rocprofv3 dispatch table.  Zero on every LoD2 kernel.  Several kernels sit at
  a ~4.8 us floor, so anything at that value is already free.

### Where the remaining decode time is, and the one open judgement call

`lod2_k_part` moves ~65 MB/layer at 624 GB/s; `lod2_k_logits_kv` ~10.9 MB/layer
at 469 GB/s; dense FA moves 134 MB/layer at 2.2 TB/s.  The state is 6x smaller
than the KV cache and we are ~4x less efficient at reading it, which is why LoD2
does not yet win despite touching far less data.  The gap is not occupancy (NS
flat), not redundancy (G sharing hurts) and not spilling (none).  What is left
that dense has and we do not is **F16**: dense moves half the bytes per element.

Storing the maintained *mean* table in F16 (means of values, O(1), 11-bit
mantissa -> ~1e-3 relative) would halve the traffic of the two largest decode
kernels.  It is not the same thing as the F16 experiment already rejected in
section 3, which converted the *sums* - those are count-scaled and lose badly.
This was deliberately **not** done: it moves the coarse branch off exact-F32
agreement with the reference, and the oracle chain at 1.07e-6 is the asset this
port is built on.  That is the user's call to make, not a silent optimization.

Everything else worth trying, in order of measured size: graph reuse (~1.0
ms/token, `can_reuse` at `src/llama-graph.cpp`; two prior attempts failed, the
note there says instrument rather than guess), `lod2_k_sim` (0.74 ms/token,
never examined), and prefill's `lod2_k_attn`, which is untouched and still the
whole prefill story - the three-contraction map in section 3 stands.

## 4c. Multi-GPU: LoD2 gets no device scaling on prefill (open)

Measured on the 27B at 32k, `-ub 4096`, prompt throughput, `HIP_VISIBLE_DEVICES`
to vary the device count:

| | 1 GPU | 8 GPU | scaling |
|---|---|---|---|
| dense | 2709.9 | 5596.0 | **2.06x** |
| LoD2  |  878.1 |  853.7 | **0.97x** |

So the 6.55x prefill gap at 8 GPUs decomposes as **3.09x extra GPU work**
(`lod2_k_attn`, section 3/4b) **times 2.12x of lost device scaling**.  At one
GPU the gap is only 3.09x, which is exactly the GPU-work ratio.  Fixing the
scaling is worth as much as the whole three-contraction rewrite.

The kernel trace agrees independently: over the prefill region the sum of
per-device busy time divided by the union of busy intervals is **1.90x for
dense and 1.00x for LoD2** - dense really does have ~2 devices computing at
once, LoD2 never has more than one.

Decode cannot benefit from this and does not: one token is one chunk, so both
dense and LoD2 measure 1.00x overlap there.  Decode's remaining gap is the
section 4b table, not this.

### Ruled out

- **Graph reuse is not the cause.**  Dense prefill is 5699.5 t/s with reuse and
  5699.0 with `LLAMA_GRAPH_REUSE_DISABLE=1` - no effect, and dense reuses
  graphs across prefill ubatches (same n_tokens) while LoD2 never does.  So the
  ~1.0 ms/token that graph rebuild costs at decode is a separate, smaller
  problem.  `can_reuse` is now behind `LLAMA_LOD2_REUSE` (1 = on, 2 = on with a
  per-decision trace) instead of `return false`, for whoever picks that up; it
  still produces wrong output when enabled, and the trace showed the function is
  not even reached in the failing case, so start there.
- **Not clock skew in the trace.**  The 1.90x/1.00x from the trace and the
  2.06x/0.97x from throughput are independent measurements that agree.

### Leading hypothesis

ggml's pipeline parallelism (`ggml_backend_sched_new(..., parallel=true)`,
n_copies = 4) overlaps consecutive `graph_compute_async` calls: during prefill
only the last ubatch needs logits, so ubatches 0..6 never force a sync and
device 0 can start ubatch k+1 while device 7 finishes ubatch k.

Dense's KV cache is append-only and each ubatch writes disjoint cells, so that
overlap is safe.  **LoD2's state is a read-modify-write of the same slots**: the
update for ubatch k+1 merges into rows that ubatch k just wrote.  The dependency
is per layer, so a correct pipeline is still possible in principle (layer 0 of
k+1 after layer 0 of k, and layer 0 lives on device 0), but nothing in the graph
expresses it - the chaining through the `state` src only orders nodes *within*
one graph.

Next step: `GGML_SCHED_DEBUG=2` on one prefill ubatch for dense and for LoD2 and
diff the split structure.  What to look for is whether LoD2 produces a split
that the scheduler has to synchronize on (a node it cannot place on the layer's
own device, or an input written with a blocking `ggml_backend_tensor_set`
mid-graph), or whether the scheduler is pipelining fine and something in
`llama_context` is draining it every ubatch.  Do not assume the answer is the
state dependency until the split dump says so - two mechanisms have already been
ruled out by measurement here.

## 4d. Why the author's Triton is far faster than lod2_k_attn

Read the kernels, do not infer this from timings.  The author's local-window
loop (`kvm_paper/model/kernels/lod_kernels.py`, ~line 1281 and ~1569):

    for local_begin in tl.range(0, local_len, BLOCK_N):
        keys   = tl.load(...)                                  # [BLOCK_N, D]
        values = tl.load(...)
        scores = SCALE * tl.dot(queries, tl.trans(keys))       # [BLOCK_M, BLOCK_N]
        block_maximum = tl.max(scores, axis=1)
        ... one rescale for the whole tile ...
        accumulator = accumulator*correction[:,None] + tl.dot(probabilities, values)

`queries` is a [BLOCK_M, D] tile; launches use BLOCK_M 16/32.  Ours:

    lod2_k_attn<<<dim3((nq + wpb - 1)/wpb, Hq), LOD2_BLOCK, ...>>>   // wpb = 4

grid.x = nq/4 with 4 warps per block, i.e. **one warp per query**.  Four
independent multipliers, all in our favour to fix:

1. **No query tiling.**  The author loads a K/V tile once and applies it to
   BLOCK_M queries; we re-stream the whole local window and the whole slot table
   for every query.  Traffic is BLOCK_M times higher (16-32x).
2. **No matrix cores.**  `tl.dot` lowers to MFMA on CDNA.  We compute each score
   as a `lod2_warp_sum` - a six-step cross-lane shuffle reduction *per (query,
   column) pair* - and each output as scalar `v_fma_f32`.
3. **Per-column online-softmax rescale.**  The author rescales once per BLOCK_N
   tile; we rescale on every column, so the accumulator carries a dependent
   exp2f chain.  This is the exact defect removed from the *decode* kernel in
   section 4b, which was worth 4.1x there.  `lod2_k_attn` still has it.
4. **Dtype.**  The author's tiles feed tl.dot as bf16/fp16 with fp32 accumulate;
   our mean table is F32 throughout.

So the answer to "why does the port not reproduce it" is not that ggml cannot:
`lod2_k_attn` is still the phase-1 correctness kernel and was never restructured.
Section 3's three-contraction map is the plan; item (1), the local window, is
where 1, 2 and 3 above all land.

Worth knowing before writing it: llama.cpp's own FA MMA fast path is disabled
under `GGML_USE_HIP` (vec fallbacks are used), so dense attention on this box is
*also* not on matrix cores.  A properly tiled LoD2 kernel would not merely catch
up with llama.cpp's dense attention here - it could pass it, since LoD2 touches
far less data.  The blocker named in section 3 stands: `ggml_flash_attn_ext`
does not return the logsumexp the three-branch merge needs.

### 4d.1 Tried: the per-tile rescale alone does NOT help prefill

`lod2_k_attn`'s local branch now scores LOD2_QT=8 columns, takes the tile
maximum and applies one correction, exactly like the author's `for local_begin
in tl.range(0, local_len, BLOCK_N)` loop.  Correctness held (oracle 36/36, both
end-to-end checks IDENTICAL).  **Prefill did not move: 864.7 -> 885.9 t/s, at
the edge of the +-2.5% noise band.**

That is a useful negative.  The same transformation was worth 4.1x on the decode
kernel, and the difference is parallelism: decode has 24 units, so a dependent
exp2f chain is exposed; prefill launches nq*Hq = 98304 query-warps, so the chain
is already hidden behind other warps.  Prefill is not latency bound.

Therefore the remaining prefill gap is **raw work**, and only the two factors
that reduce the operation count can touch it:

- **query tiling** - a K/V tile applied to BLOCK_M queries instead of re-read
  per query, and
- **matrix cores** - `tl.dot` producing a [BLOCK_M, BLOCK_N] score tile instead
  of one `lod2_wsum` (a 6-step cross-lane reduction) per (query, column).

There is no cheap version of these; they need the real tiled kernel.  Do not
spend time on further peephole work in `lod2_k_attn` - this measurement says it
will not pay.

Also fixed here: the launcher sized the grid with `LOD2_BLOCK/LOD2_LANES` (4)
while the kernel indexes warps with `blockDim.x/WARP_SIZE` (8), so twice the
needed blocks were launched and half the warps ran routing and coarse with
`active == false` before hitting the guard.  That guard used to sit after a
collective staging step that has since been removed.  Worth ~2.5%.

### 4d.2 Added: lod2_k_local, a tiled local branch (LLAMA_LOD2_TILED=1)

First real port of the Triton structure, added **beside** the existing kernel
rather than replacing it, and off by default.

    27B prefill, tiled=0 :  866.6 t/s
    27B prefill, tiled=1 : 1074.1 t/s     (+24%)

    TILED refinement all-open vs dense : IDENTICAL
    TILED default vs dense             : IDENTICAL

Shape: a block owns LOD2_TM=16 queries of one query head and walks the window in
LOD2_TN=16 column tiles with Q, K and V staged in LDS (~50 KB at D=Dv=256, which
is why the launcher gates on `lds <= 64*1024`).  It writes the local branch's
(m, d, acc) to scratch and `lod2_k_attn` loads them instead of running its own
local loop, so the three-branch logsumexp merge is untouched - that is what kept
this verifiable in one step.

Two design points worth keeping if this is rewritten:

- Thread t owns query `m = t/LOD2_TN` in *both* the score mapping (`n = t%TN`)
  and the value mapping (`dv = t%TN + TN*e`).  Because it is the same m, the
  per-query softmax state (m_run, d_run, correction) stays in registers
  replicated across the TN threads of a row - no LDS, no barrier for it.
- Staging is transposed where the access wants it: `ks` is [D][TN] so the TN
  threads of a score row read consecutive floats, `vs` is [TN][Dv] likewise.

Still per-query inside `lod2_k_attn`: the coarse and routed branches.  Coarse is
the next largest (s_len * Dv per query) and the same treatment applies.

**Not done: MFMA.**  Scores are still one thread's serial dot over LDS, which is
~1.8x fewer wave-instructions per score than the six-step cross-lane reduction
it replaced, but `tl.dot` lowers to matrix cores and this does not.  That is the
remaining multiple, and the kernel is now shaped so it can be dropped in as a
body swap (16x16x16 tiles already match `__builtin_amdgcn_mfma_f32_16x16x16f16`;
K/V would move to f16 in LDS, halving the staging budget as a bonus).

### 4d.3 Added: lod2_k_coarse, and where the LDS road ends

Same treatment as 4d.2, applied to the branch the profile named next.  Both are
bits of LLAMA_LOD2_TILED now - 1 local, 2 coarse, 3 both - so either can be
measured alone.

    27B prefill, 1 GPU        TILED=0  896.4    dense 2712.1
                              TILED=1 1088.1    local only
                              TILED=2 1027.4    coarse only
                              TILED=3 1306.0    +45.7%

    27B prefill, 8 GPU        TILED=0  888.7 -> TILED=3 1249.3   +40.6%
    27B generation            58.9 either way (decode uses the split path)

    oracle 36/36 (CPU + 8 ROCm), output max_abs 2.146e-06
    TILED=1/2/3 refinement all-open vs dense : IDENTICAL
    TILED=1/2/3 default vs dense             : IDENTICAL

lod2_k_coarse mirrors lod2_k_local: a block owns TM queries of one query head
and walks the slot table in LOD2_TN columns, staging each column tile's value
means once for all TM queries.  It needs no Q or K staging at all - the logits
come from the routing GEMM - so its shared memory is just the value tile, and
it is the cheaper of the two kernels by 5x.

The routing scan is replicated verbatim inside it rather than passed over from
lod2_k_attn.  It reads two floats per slot against the coarse branch's Dv, so
it is noise next to what the tiling saves, and running the identical scan in
the identical order keeps both kernels' route lists bit-identical - which is
what let this be verified in one step without touching the merge.

What that did to the profile (27B, 32k, kernel time summed over 8 devices):

                        before      after
    lod2_k_attn        17949.1      887.8     (routing + routed + merge only)
    lod2_k_local             -     6025.4
    lod2_k_coarse            -     1169.9
    attention total    17949.1     8083.1     2.22x

lod2_k_attn is no longer the rate limiter.  lod2_k_local is, at 74% of the
attention time.

**Three negative results, all worth not repeating.**

1. *Q and K staged as [TM][DP] / [TN][DP] rows, read four floats at a time.*
   Cuts the score loop's LDS instruction count by four and makes the staging
   loads coalesced instead of strided.  Measured **13.9% slower** (1088 ->
   936).  Reverted.
2. *Value loop interchanged to column-outer / dimension-inner*, so ps[m][u] is
   loaded once per column instead of once per (column, dimension): two LDS
   reads per FMA down to 17/16, and bit-identical output.  Measured **2.3%
   slower** on the coarse kernel.  Reverted.
3. *Wider tiles.*  The coarse kernel's shared memory is dominated by the value
   tile, which does not depend on TM, so TM can go to 64 freely.  16 -> 1014.6,
   32 -> 1026.6, 64 -> 1026.0 t/s.  **+1.2% and then flat.**  Default is 32.
   (The local kernel also stages Q, and D*32 floats of it no longer fit beside
   the K and value tiles, so 16 is all that is reachable there.)

4. *Four independent partial sums in the score dot*, to break what looks like a
   serial dependent FMA chain D long - the same defect that was worth 4.1x on
   the decode kernel in 4b.  **No effect** (1088.1 -> 1082.0).  The compiler
   already schedules across it.

Read together these say the same thing: **reuse is saturated and the kernels are
not issue bound.**  Everything the tiling bought - the +24% in 4d.2 and the
+45.7% here - came from reading a column once per TM queries instead of once per
query.  That well is now dry, and neither instruction count nor more reuse moves
the needle.

**But the shared memory is now the problem, by volume rather than by layout.**
From the dispatch records and the kernel symbol table:

    kernel           lds/wg    wg   blk/CU   wave/CU   wave/SIMD   vgpr
    lod2_k_local      50176   256        1         4         1.0     96
    lod2_k_coarse     17472   256        3        12         3.0
    lod2_k_sim         2560   256        8        32         8.0

A CU has 64 KB of LDS, so lod2_k_local's 50176 bytes admit exactly one
workgroup, which is four wave64s across four SIMDs: **one wave per SIMD, no
latency hiding at all.**  Its 96 VGPRs would allow five.  The staging that
bought the reuse is what is now starving the kernel, and registers are not the
binding constraint - shared memory is, by 5x.

Getting to two workgroups per CU needs the footprint under 32768 bytes, and
with D = Dv = 256 that is not reachable by retuning TM and TN: the budget is
    256*TM + 256*TN + 256*TN + TM*TN <= 8192  with TM*TN = 256
whose best case is TM + 2TN = 45 against a limit of 31.  It needs the K tile
staged in D chunks (and probably the value tile in Dv chunks), e.g.
    qs [D][16] 16384 + ks [64][16] 4096 + vs [16][128] 8192 + ps 1024 = 29696
which is two workgroups per CU, i.e. 2 waves per SIMD.  That is a real
restructure of the tile loop - partial scores accumulated across D chunks
before the row maximum can be taken - and it is the next thing to try.

Worth knowing before starting: at 1 wave/SIMD the kernel already moves 14.6 TB/s
of LDS traffic against a device peak near 81.7, so it is at 18% of the shared
memory roofline.  Doubling occupancy should approach 2x and no more.  That is
worth having but it does not close the gap below.

**What is ultimately left is arithmetic throughput, and here is the size of it.**

    dense flash_attn_ext_f16                            72.7 TFLOPS
    lod2_k_local   2.07e11 FLOP/dispatch / 56.6 ms  =    3.7 TFLOPS

The lod2_k_local figure is counted from the dispatch records rather than
modelled: all 96 full-ubatch dispatches carry grid 65536 x 24 and land within
42.8-59.9 us of each other, which is itself the evidence that the local window
stays inside the ubatch (l0 = q0).  Block b then walks b+1 column tiles, so a
dispatch is sum(b+1, b<256) * 24 = 789504 tile iterations of 131072 FMA each.
That is 4.5% of the 81.7 TFLOPS f32 vector peak, and 20x off dense.  llama.cpp's own attention runs at essentially peak on this box while
ours runs at 6% of it, and no shared-memory arrangement closes that - the score
loop is scalar f32 v_fma out of LDS, one thread per (query, column).  The two
ways out, in increasing order of what they cost the verification chain:

- packed f32 (v_pk_fma_f32) - 2x, exact, no numerics change;
- MFMA - the tile shape is already 16x16, so __builtin_amdgcn_mfma_f32_16x16x4f32
  is a body swap that keeps f32 operands, and the f16 variant is 8x more but
  moves the local branch off exact-f32 agreement.  That last one is a judgement
  call for the author, not a free optimisation: see the same open question about
  the mean table at the end of section 4b.

Standing after all of this, 1 GPU, 27B/32k prefill:

    ours 1306.0   author's Triton 2000   our dense 2712.1
    gap to Triton 1.53x   (was 2.24x)
    gap to dense  2.08x   (was 3.03x)

### 4d.4 Narrow the staging, not the arithmetic

The occupancy finding above has a fix that costs nothing in accuracy, because
**K and V already live as F16 in the cache**.  lod2_get widens them to F32 on
load and the old kernel then stored that widened copy in shared memory, so
narrowing it back is exactly undone at the point of use: every float that
reaches the arithmetic is bit-identical to the F32-staged path.  ps is also
folded into the K tile's bytes - the scores are finished with K before any
probability is written, so the two are never live together - which is the last
1024 bytes that stood in the way.

    lod2_k_local<32, true>   lds 49152  wg 512  ->  2 waves/SIMD   (was 1.0)
    lod2_k_coarse<32>        lds 18496  wg 512  ->  6 waves/SIMD   (was 3.0)

    lod2_k_local   6025.4 -> 3389.7 ms   1.78x
    lod2_k_coarse  1169.9 ->  864.1 ms

Note which knob actually moved: halving the two big buffers let the launcher's
own "largest row count that fits" rule pick TM=32 instead of 16, and a 512
thread workgroup is 8 waves.  Same 2 waves/SIMD as two 256-thread workgroups
would have given, with twice the reuse.

Then packed f32, per the author's decision to keep MFMA out on portability
grounds.  Q is staged as [D/2][TM] of float2 and K as [D/2][TN] of half2, so
the score loop reads two elements per LDS instruction and the two multiplies it
then does are independent and adjacent.  **v_pk_fma_f32 is emitted - 3280 of
them in the object - and it is worth 1.3%.**  Confirming the instruction matters
here: the mechanism works and the gain is still ~1%, which is the fifth
independent measurement saying this kernel is bound by shared-memory access and
not by arithmetic issue.

    27B prefill, 1 GPU     TILED=0  895.2      dense 2704.5
                           TILED=1 1225.7      local only
                           TILED=2 1026.8      coarse only
                           TILED=3 1511.8      +68.9% over untiled

    27B prefill, 8 GPU     887.6 -> 1489.2
    27B generation         59.2 / 59.1 (decode path untouched)
    oracle 36/36, all six refinement and default checks IDENTICAL

    gap to Triton 1.32x   (2.24x at the start of this work)
    gap to dense  1.79x   (3.03x)

Where the remaining shared-memory pressure is: Q is the one tile still staged
F32, and at 32768 bytes for TM=32 it is now two thirds of the footprint.
Narrowing it is *not* free - Q arrives F32 from the graph, so half would cost
real precision in the score - which leaves D-chunking it as the way to 4
waves/SIMD.  ds_read_b32 still outnumbers ds_read_b64 in the object by 100:1,
so the float2 Q reads are being split; making that alignment provable is the
one cheap thing left to try before the restructure.

### 4e. Prefill does not scale, and the attention kernels are not why

Measured on one GPU, 27B, same build (TILED=3):

              dense     lod2     gap
     6k      2671.9   1918.6    1.39x
    32k      2709.1   1515.2    1.79x
   100k      2188.5    786.9    2.78x

The gap WIDENS with context.  That is backwards for a sparse-attention method
and it is the single most important fact about this port's prefill.

Per-kernel, 32k -> 100k on one GPU.  Token count grows 3.2x, and mul_mat_q
(3.21x) and gated_delta_net (3.22x) give the O(N) baseline to read against:

    lod2_k_sim      1674.0 -> 30516.7   18.23x    5.7x super-linear
    lod2_k_merge    1049.4 -> 16171.1   15.41x    4.8x
    lod2_k_pages     800.9 -> 10370.8   12.95x    4.0x
    lod2_k_split     225.3 ->  2240.1    9.94x    3.1x
    lod2_k_coarse    858.3 ->  5641.0    6.57x    2.1x
    lod2_k_attn      889.3 ->  5364.8    6.03x    1.9x
    lod2_k_local    3124.2 -> 10527.8    3.37x    linear, healthy

At 100k the state maintenance - sim, merge, pages, split, dest - is **64% of
the entire prefill** (60428 of 93960 ms).  All three attention kernels together
are 23%.  Every optimisation in 4d went into that 23%.

Where the super-linearity comes from, as far as this goes: lod2_k_sim's grid is
*identical* at both lengths - 327680 work items, i.e. M = 320 columns per update
times Hkv = 4 - so the kernel is not getting bigger.  It is being launched more:

    32k :  1264 launches, 1.25 ms each
    100k: 12496 launches, 2.41 ms each

The per-launch 1.9x is expected and benign: sim compares M columns against
s_len slots, and s_len grows with context.  **The 9.9x launch count for a 3.2x
token increase is not explained by tokens/M and is the thing to chase first.**
ggml_cuda_lod2_update launches sim exactly once per call, so this is the number
of update calls, not anything inside the kernel.

Consequences for planning:

- Prefill cannot be fixed from the CUDA attention kernels.  They are 23% at
  100k and the healthiest part of the profile.
- At 32k, dense attention is only 12.5% of dense's own prefill, so even a free
  LoD2 attention could not beat dense there - the ceiling is 1.14x.  LoD2's
  prefill case only exists at long context, which is exactly where this port
  currently degrades fastest.
- lod2_k_sim alone at 100k (30.5 s) is larger than every attention kernel in
  the model put together.

### 4e.1 The scaling defect, narrowed but NOT yet found

The author's position, and it is correct: LoD2 is sublinear attention, so a
correct implementation cannot produce O(N^2) compute.  4e's numbers are
therefore a defect to find, not a cost to tune.

What is established:

- **Slots are correctly sublinear.**  desired_slots = state_growth*sqrt(ctx)
  with state_growth 16: 2896 slots at 32k, 5060 at 100k.
- **The update step is a constant**, prefill_state_update = 1280, and
  lod2_k_sim's grid confirms it - 327680 work items = 1280 columns x LOD2_BLOCK,
  identical at both context lengths.
- Therefore lod2_k_sim should be launched **N/1280 times per layer**, and its
  cost should total O(N * sqrt(N)).

What is measured:

                       expected (N/1280 * 16 layers)   actual    excess
    32k (27648 tok)                  346                1264      3.65x
    100k                            1250               12496     10.0x

The excess itself grows with N, which is the whole scaling defect.

**A mechanism was proposed and is refuted.**  "coverage does not persist across
ubatches, so every ubatch re-clusters the whole prefix" reproduces the measured
counts almost exactly (1075 vs 1264, 15360 vs 12496).  But the code commits it:
llm_graph_input_attn_lod2 carries coverage_post, build_attn writes it at
llama-graph.cpp:3564, and llama-context applies it via set_lod2_state at
llama-graph.cpp:3379.  The numeric agreement is a coincidence and the mechanism
is not this.  Do not spend time re-deriving it.

**Next step, and it is cheap:** instrument llama_lod2_plan_prefill to log
(il, p0, coverage, target, blocks.size()) for the first few ubatches of a 32k
prefill.  If blocks.size() is ~3 per layer per ubatch as the arithmetic says,
the extra launches are not coming from the planner and the search moves to how
many times build_attn runs per ubatch.  If it is larger, the planner's coverage
or target is wrong on entry and the answer is right there.

Also worth checking in the same pass: lod2_k_attn's 144 calls at 32k does not
factor as 16 layers x 7 ubatches, so the per-ubatch layer count may itself not
be what this analysis assumed.

### 4e.2 FOUND AND FIXED: a no-op seq_rm was destroying the state every ubatch

The author was right that a sublinear method cannot produce O(N^2), and it was
a defect.  Instrumenting llama_lod2_plan_prefill showed coverage entering every
ubatch as **zero**:

    p0= 4096  cov=0 tgt= 3328  blocks= 4
    p0= 8192  cov=0 tgt= 7424  blocks= 7
    p0=12288  cov=0 tgt=11520  blocks=10
    ...       (92 blocks per layer against the 22 the plan needs)

The commit path was not at fault - build_attn published cov_post=3328 correctly
and set_lod2_state stored it on the right cache object and stream.  A clear was
landing immediately after every store, and it did not go through the setter:

    LOD2SET   strm=0 cov=3328
    LOD2CLEAR strm=0 was cov=3328
    LOD2CLEAR_FROM seq_rm seq=0 p0=4096 p1=-1 used=4096

**llama.cpp issues seq_rm(seq, n_past, -1) between ubatches as a tidy-up, and
with p0 == used it removes nothing at all.**  llama_kv_cache::seq_rm cleared the
whole LoD2 state on any suffix removal, so a call that took back zero cells threw
away the entire clustering, and the next ubatch rebuilt it from position zero.
Output stayed correct - a rebuild from zero lands on the same state - which is
why every end-to-end check passed throughout.

Fix: only clear when the removal actually takes back columns the state absorbed.
The state covers [0, coverage), a suffix removal touches [p0, used), so the test
is `p0 < coverage` per stream (llama-kv-cache.cpp, seq_rm).

    27B prefill, 1 GPU        before      after
      6k                      1918.6     2030.2     dense 2695.5   1.33x
     32k                      1511.8     1825.3     dense 2707.7   1.48x
    100k                       786.9     1551.5     dense 2187.3   1.41x

    27B prefill 32k, 8 GPU    1489.2 -> 1785.3
    generation                59.1 -> 59.8
    oracle 36/36, all six refinement and default checks IDENTICAL

The gap no longer widens with context - it is flat near 1.4x and improves from
32k to 100k, which is the shape a sublinear method should have.  Note also that
TILED=0 moved (896 -> 995), because this defect was never in the attention
kernels at all.

Re-adding the instrumentation, if it is needed again: an env-gated
fprintf(stderr) of (il, n_tokens, p0, p1, coverage, target, slots,
blocks.size()) around the llama_lod2_plan_prefill call in build_attn, plus one
in clear_lod2_state_stream.  LLAMA_LOG_INFO does not reach the console under
llama-cli's default log setup - that cost one build cycle.

### 4f. lod2_k_local_ds: the dimension-split local branch

The tiled local kernel (4d.3/4d.4) gives one thread one (query, column) pair.
Its dot product therefore reads D floats of Q and D of K out of shared memory to
produce 2D flops - one element moved per flop, no reuse inside the thread - and
its value pass reads the probability back out of shared memory once per (column,
dimension).  Counted per thread per column tile that is 768 shared-memory
instructions for 1024 flops.  Everything tried against the instruction mix
(float4 rows, split accumulators, packed f32) moved it by at most 1.3% because
the ceiling was the shared-memory pipe, not the arithmetic.

lod2_k_local_ds splits the head dimension across the LOD2_TN lanes of a row
instead of the columns.  Lane n owns dims [n*DP, (n+1)*DP) of its queries for the
whole window, which changes three things at once:

  - Q lives in registers, loaded once from global and never re-read;
  - a lane reads DP contiguous elements of K per column, which merge into wide
    ds_read instead of DP scalar ones;
  - the probabilities stay in registers, so the ps round trip through shared
    memory and the two barriers around it disappear.

The staged tile is [LOD2_TN][D] and [LOD2_TN][Dv], so the shared-memory
footprint is 16 KB and no longer depends on the block width at all.

Measured, 27B / 32k / 1 GPU, TILED=1, rocprofv3, 144 dispatches:

    old lod2_k_local<32, true>          3130.0 ms   lds 49152
    lod2_k_local_ds<16, 1, 16, true>    2532.1 ms   lds 16384
    lod2_k_local_ds<16, 2, 16, true>    2525.0 ms   lds 16384   <- default
    lod2_k_local_ds<32, 1, 16, true>    3256.6 ms
    lod2_k_local_ds<32, 2, 16, true>    2676.4 ms
    lod2_k_local_ds<64, ...>            2871.4 ms   scratch 16  (spills)

1.24x on the kernel; end to end 27B/32k/1 GPU TILED=3 1822.7 -> 1897.1 t/s, 8 GPU
1734.3 -> 1865.0.  Verified the usual way: refinement all-open vs dense IDENTICAL
for TILED=1/2/3, default vs dense IDENTICAL, oracle 36/36.  The kernel is NOT
bit-identical to the old one - the dot product is summed per lane and then across
lanes where the old one sums it serially in one lane - and that is fine, because
the merge that consumes it is a logsumexp over independent branches.

Three things this measurement settled, all of them worth not re-deriving:

  1. D as a kernel argument defeats the load widening.  The first version
     indexed the tile with the runtime D, so the compiler could not see that a
     lane's slice starts on a DP boundary and emitted one ds_read per element;
     that version measured 1778 t/s end to end, i.e. WORSE than the old kernel.
     Writing the stride as the compile-time DP*LOD2_TN (the two are equal by
     construction) took it to 1899.  Same arithmetic, same layout, 6.8%.
  2. The block width is the knob, not the reuse.  RG 16 (256 threads) beats
     RG 32 by 29% at RM 1, and RG 64 spills.  RG 16 also does twice the staging
     traffic of RG 32 and still wins, so this kernel is not staging bound.
  3. Register blocking over queries is worth almost nothing here.  RM 2 halves
     the f16->f32 widening per query and measured 2532 -> 2525 ms, 0.3%.  It is
     kept as the default only because it halves the block count, and therefore
     the global K/V re-reads, for free.

What the disassembly says is left (gfx942, per column tile, RG16/RM1):

    v_cvt_f32_f16   512    ds_read2_b32    128
    v_pk_fma_f32    232    ds_bpermute_b32  64
    v_add_f32       128    v_exp            17
    v_pk_mul_f32     32

1990 instructions for 1024 flops, i.e. 0.51 flops per instruction.  The
conversions are the largest single class and RM 2 proved they are not the
binding constraint, so the next thing to try is not the instruction mix.  Note
that llama.cpp's own flash attention reaches 72.7 TFLOPS on this device with no
matrix cores either - it uses half2 arithmetic throughout and never widens.

### 4f.1 The binding constraint was LDS bank conflicts, not conversions

4f left the kernel at 0.51 flops per instruction with 512 v_cvt_f32_f16 per
column tile as the largest single class.  Removing them looked like the obvious
next move, and it is available for free: staging K and V as f32 instead of f16
deletes the widening on the read side and the narrowing on the store side at
once, because lod2_get already returns f32.  The f32 tile is twice the bytes, so
the column tile CT was halved to hold the footprint and the occupancy fixed.
CT is a free parameter in this kernel - the lanes carry the head dimension, not
columns - which is what makes the trade available at all.

It measured 1.70x SLOWER (2542.3 -> 4310.1 ms) at identical shared memory,
identical occupancy and 512 fewer instructions.

That is the measurement that identifies the constraint.  Fewer instructions and
more shared-memory bytes, everything else held: the kernel is bound by the LDS
pipe.  And the cost was not bandwidth but conflicts.  A lane owning a contiguous
slice [n*DP, (n+1)*DP) starts every DP*sizeof bytes; at DP 16 halves that is 32
bytes, so lane n reads dwords 8n and 8n+1 and the sixteen lanes of a row cover
only eight banks - a four-way conflict on every K and V read.  In f32 the stride
doubles and so does the conflict, which is the 1.70x.

The fix is to stripe the slice instead of taking it contiguously: lane n owns
runs of LOD2_CH = 4 dimensions, run c starting at c*LOD2_CH*LOD2_TN + n*LOD2_CH.
Lane n then sits at dword 2n within a run, so the sixteen lanes cover
thirty-two consecutive dwords - one per bank - at the same eight bytes per lane
and the same instruction count.  Q, the accumulators and the output write all
follow the same mapping; the consumer reads lacc by index, so nothing downstream
changes.

Local branch alone, 27B / 32k / 1 GPU, 144 dispatches:

                        contiguous   striped
    f16 tile, CT 16        2542.3     2358.7
    f16 tile, CT  8        2579.7     2336.3
    f32 tile, CT 16        4583.2     2490.6
    f32 tile, CT  8        4310.1     1961.4   <- default

Striping alone is worth 7% on the f16 tile.  What it really buys is that the
conversion-free tile becomes affordable: f32 with CT 8 goes from the worst
variant to the best, 1.30x over the previous default and 1.60x over the tiled
kernel this section started from (3130.0 ms).

Two conclusions worth keeping.  First, an optimisation that is right on paper can
be hidden entirely by a layout defect - the f32 tile was always the better idea
and measured terrible for a reason that had nothing to do with it.  Second,
"which instruction class is largest" is not the same question as "what is the
constraint": the 512 conversions were the largest class and were never the limit,
and the two experiments that pointed at that (RM 2 worth 0.3%, f32 worth -70%)
were both cheap.

End to end, 27B / 32k, TILED=3: 1 GPU 1895.2 -> 1973.5, 8 GPU 3979.8 -> 4112.0
t/s.  Verified: refinement all-open vs dense IDENTICAL for TILED=1/2/3, default
vs dense IDENTICAL, oracle 36/36, generation unchanged at 60.0 t/s.

### 4f.2 Overhead reduction: three negative results, and RM 2 pinned

After striping, the body is 1680 instructions for 1024 flops, of which only 451
are arithmetic and 128 move data; the other ~1100 are address arithmetic, exec
mask manipulation, register moves and s_waitcnt.  That looks like an obvious
place to work.  Three attempts, all measured on the local branch alone, 27B /
32k / 1 GPU:

  1. Packing the score FMAs.  The score is the only unpacked chain - 240
     v_fmac_f32 against the value pass's 129 v_pk_fma_f32 - because it
     accumulates both halves of a pair into one scalar.  Splitting it into two
     partial sums, which is exactly the shape the value pass has, produced a
     disassembly identical BYTE FOR BYTE and 1961.4 -> 1960.1 ms.  The compiler
     already reassociates this and declines to pack it for a reason that is not
     the source form.  The change is kept only because it states the intent.
  2. RM 1.  The hypothesis was that RM existed only to amortise the f16->f32
     widening, which the f32 tile has since removed, so RM 2 would now be pure
     register pressure.  Wrong: 2651.2 ms against 1960.1, i.e. RM 2 is worth
     26%.  What RM actually amortises is the LDS read - a thread's 64
     ds_read2_b64 serve RM queries - and that still matters.
  3. RM 4.  Follows directly from 2, and does not hold: 2102.6 ms, with
     arch_vgpr 24 -> 124.  The register pressure costs more than the halved LDS
     traffic buys.  RM 2 is the optimum of the ladder 2651 / 1961 / 2103.

The pattern across all three, and across 4f's RM experiment before them, is that
this kernel no longer responds to instruction count.  It is in a balanced regime:
LDS traffic matters (RM 2 is worth 26%), register pressure matters (RM 4 costs
7%), and raw instruction count does not.  Nobody should expect the remaining
identified lever - splitting the tail tile so the 168 exec-mask and 78 cndmask
instructions leave the main body - to behave differently; it is worth trying, but
it is 15% of a kernel that is 14% of LoD2's prefill, so about 1.7% end to end.

Where the evidence says to work instead: lod2_k_attn and lod2_k_coarse together
are 12% of prefill and have had none of this attention, and the state
maintenance is another 8%.  Neither has been through a single layout pass.

### 4f.3 The same three findings applied to lod2_k_coarse

The coarse kernel's value loop had exactly the local branch's old disease: 512
shared-memory reads per column tile for 512 flops, made of 256 reads of ps - one
per (dimension, slot), all sixteen lanes of a row wanting the same float - and
256 reads of vs one dword wide.  Three changes, all transplanted:

  - NACC = Dv/LOD2_TN as a template parameter, so the tile stride is compile
    time and a lane's slice is provably aligned (the compile-time stride
    finding from 4f);
  - the sixteen probabilities read into registers once per tile instead of once
    per (dimension, slot) - four broadcast reads against 256;
  - value dimensions taken in striped pairs, lane n at dword 2n within a run, so
    a row's sixteen lanes cover thirty-two consecutive dwords, one per bank, at
    eight bytes each (the striping finding from 4f.1).  The form it replaces was
    lane-strided: conflict free already, but one dword per lane per instruction,
    so twice the instructions.

Local branch untouched, 27B / 32k / 1 GPU, 96 dispatches, by row count:

    lod2_k_coarse<16>   1055.0 ms
    lod2_k_coarse<32>    779.5 ms   <- default, was 870.0
    lod2_k_coarse<64>    848.5 ms

10.4% on the kernel.  End to end that is 0.6% and it does not clear the noise
floor: 27B/32k TILED=3, 1 GPU 1973.5 -> 1987.5 t/s, 8 GPU 4112.0 -> 4040.8 with
dense moving 5660.9 -> 5689.7 in the same runs.  Verified: refinement all-open
vs dense IDENTICAL for TILED=1/2/3, default vs dense IDENTICAL, oracle 36/36,
generation unchanged at 59.8 t/s.

Not attempted, and the honest next targets in order: lod2_k_attn is now the
largest LoD2 kernel after the local branch at 897.3 ms and has had no layout
work at all, and lod2_k_sim at 564.9 ms is the largest of the state maintenance
kernels.  Together with merge and pages they are about 8% of prefill.

The expectation to carry into that work is the one 4f.2 established: these
kernels respond to shared-memory traffic and register pressure, not to
instruction count, and a 10% kernel win is worth well under 1% end to end.  The
per-column efficiency gap to dense is not going to close one kernel at a time.

### 4g. There is no O(N^2) left anywhere

Checked twice, statically and by measurement.

Statically, the only super-linear term in the whole path is the slot count
s_len = 16*sqrt(N):

    lod2_k_sim      M*s_len*D  per update chunk, N/M chunks   -> N*sqrt(N)
    lod2_k_merge    s_len*M    per update chunk               -> N*sqrt(N)
    lod2_k_pages    s_len*M    per update chunk               -> N*sqrt(N)
    lod2_k_means    s_len*D    per update chunk               -> sqrt(N)*N/M
    lod2_k_coarse   nq*s_len*Dv per ubatch                    -> N*sqrt(N)
    routing GEMM    nq*s_len*D  per ubatch                    -> N*sqrt(N)
    lod2_k_split    M*log^2 M  per update chunk               -> N
    lod2_k_dest     M*n_append*D, n_append -> 0 as sqrt        -> sub-linear
    lod2_k_local    nq*W*D, W = prefill_local                 -> N
    lod2_k_attn     nq*routes*pagesize*D                      -> N

The local window W is bounded by construction and not by luck: build_attn sets
l0 = coverage after emit_updates, and llama_lod2_plan's `while (coverage <
target)` runs unconditionally to target = local_begin(p0) = p0 - lookback.  There
is no path on which the state fails to reach the window start and the exact
branch has to walk further back.

By measurement, 27B / 1 GPU / TILED=3, rocprofv3, 32k against 100k.  Token ratio
3.12, so linear is 3.12, N^1.5 is 5.52 and N^2 would be 9.77:

    lod2_k_local_ds   2532.2 ->  8534.9   3.37   (linear)
    lod2_k_coarse      870.0 ->  5698.1   6.55
    lod2_k_attn        911.6 ->  5465.7   6.00
    lod2_k_sim         565.7 ->  3608.1   6.38
    lod2_k_merge       297.9 ->  1619.0   5.43
    lod2_k_pages       216.7 ->  1031.7   4.76
    lod2_k_split        56.6 ->   184.1   3.25   (linear)
    lod2_k_dest         36.3 ->    76.3   2.10   (sub-linear)
    mul_mat_q         6618.0 -> 21270.9   3.21   (reference: linear)
    total GPU        14387.0 -> 55017.0   3.82   -> overall N^1.13

Every LoD2 kernel lands on either the linear reference or the N^1.5 line.
Nothing is within reach of 9.77.  One row looks alarming and is not: a rocBLAS
Cijk_ kernel reports 9.39x, but rocBLAS picks a different kernel per shape - the
routing GEMM's Cijk_ rows summed are 86.6 -> 530.2 ms, 6.1x, which is N^1.5 as
the table above predicts.

The residual gap to dense is therefore entirely a constant factor, and it does
not grow with context.

### 4h. Multi-GPU prefill did not scale, and graph REUSE was not the reason

Symptom: dense prefill scales 1 -> 2 -> 8 GPUs as 2703 / 3996 / 5685 t/s, while
LoD2 sat at 1900 / 1895 / 1861 - exactly 1.00x, i.e. layer split with no pipeline
overlap at all.

The natural suspicion is llm_graph_input_attn_lod2::can_reuse, which returns
false unconditionally (LLAMA_LOD2_REUSE, section 4c).  That is NOT the cause.
Graph reuse saves a host-side rebuild; it is not what enables pipelining.  What
enables pipelining is that ggml_gallocr can keep the previous allocation.  When
it cannot, ggml_backend_sched_alloc_splits takes this path:

    // the re-allocation may cause the split inputs to be moved to a different
    // address
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }

That is a full drain of every device, and LoD2 was hitting it on every ubatch.
Proof, 27B / 32k / 2 GPUs, GGML_SCHED_DEBUG_REALLOC=2 (abort on any
reallocation): dense finishes the whole prompt clean, LoD2 aborts immediately.

ggml_gallocr_needs_realloc fires on two things (ggml-alloc.c:1008): a change in
node or leaf count, and any tensor that outgrows what graph_reserve sized.  LoD2
was doing both.

  1. The number of ggml_lod2_update nodes varied per ubatch.  The plan advances
     coverage by one ubatch in prefill_state_update steps, and 4096/1280 is not
     an integer, so a ubatch emits three or four blocks depending on where the
     schedule lands - times 16 layers.
     Fix: llama_lod2_params::prefill_blocks_max() and a pad in build_attn.  The
     padding blocks are empty (p0 == p1) and both backends return immediately on
     M == 0.  The bound is taken from p.prefill_chunk, not from this ubatch's
     token count - an ubatch advances coverage by however many tokens the
     PREVIOUS one carried, so a short tail ubatch still needs a full one's worth
     of blocks.  Getting that wrong is an assert, and it fired on the first try.
  2. The routing GEMM view was sized by the live slot count, which grows every
     ubatch.  graph_reserve runs with slots == 0, so the very first real ubatch
     already outgrew the reservation and every later one outgrew the last.
     Fix: take the view at the full slot capacity (s_mn->ne[1]).  Only the first
     s_len columns are ever read; every consumer already takes the slot count as
     an op parameter and the row stride as lgt->ne[0].

Result, 27B / 32k prefill, TILED=3:

                 1 GPU     2 GPU     8 GPU
    dense        2703.3    3991.7    5657.9     (1.00 / 1.48 / 2.09)
    lod2 before  1900.3    1894.5    1860.9     (1.00 / 1.00 / 0.98)
    lod2 after   1894.8    2797.1    3971.9     (1.00 / 1.48 / 2.10)

LoD2 now scales exactly as dense does.  8 GPU prefill 1860.9 -> 3979.8 t/s.
Verified: refinement all-open vs dense IDENTICAL for TILED=1/2/3, default vs
dense IDENTICAL, oracle 36/36, generation unchanged at 59.9 t/s.

One reallocation still happens once at the start (the first real graph differs
from the reserved one).  That is a one-off and costs nothing; if it were still
per-ubatch the scaling above could not exist.

The lesson worth keeping: for pipeline parallelism the graph does not have to be
REUSABLE, it has to be ALLOCATION-STABLE.  Those are different properties, and
only the second one was in the way here.

## 4i. Parallel sequences (-np) and a quantized KV cache

Both were reported as unsupported and both now work.  Neither needed much: the
work was mostly finding out that my own guards, not the implementation, were the
obstacle.

### -np

The refusal was an explicit guard of mine in llama_context, left over from an
early phase:

    if (cparams.n_seq_max > 1) {
        throw std::runtime_error("LoD2 supports a single sequence in this phase - use -np 1");

Everything downstream had been per-stream all along: s_kv, s_mn, p_kv, p_idx,
s_pg and meta all carry n_stream as their last dimension, every
llama_kv_cache_context accessor goes through get_stream(), lod2_coverage and
lod2_slots are per-stream vectors, and seq_rm already clears per stream.

What LoD2 actually needs is that the archive's cell index equals the position it
reads by.  One stream per sequence gives exactly that - each sequence owns a
contiguous append-only stream whose head is its length.  A unified cache
interleaves sequences in one cell space and breaks the identity.  So the guard
became a guard on that, and only that:

    if (cparams.n_seq_max > 1 && params.kv_unified) {
        throw std::runtime_error("LoD2 needs one KV stream per sequence: "
                "run without --kv-unified, or use -np 1");

kv_unified defaults to false, so -np N works out of the box.  Mixed-stream
ubatches - several slots batched together - still take the dense path via the
existing nullptr return in build_attn_inp_lod2, which is coherent because the
archive holds exactly what dense would hold; they simply get no LoD2 benefit.

Verified with llama-parallel, 2B, -ns 4, all-open refinement against dense:

    np=1  IDENTICAL      np=2  IDENTICAL      np=4  DIFFERS

The np=4 row is NOT a LoD2 defect, and the control says so: dense compared
against itself gives np=1 vs np=2 IDENTICAL and np=1 vs np=4 DIFFERS.
llama-parallel is not reproducible at np=4 - the batching order changes and the
numerics with it - and LoD2 reproduces that pattern exactly.  Always run the
dense-vs-dense control before reading a -np diff as a failure.

### -ctk / -ctv q8_0

This one aborted, and my first explanation of it was wrong.  It is not a silent
fallback to the CPU: lod2_type_code rejected the type, ggml_cuda_lod2_supported
returned false, and the scheduler then hit

    ggml-backend.cpp:933: pre-allocated tensor (cache_lod2_s_kv_l3 (view) (view))
    in a buffer (ROCm0) that cannot run the operation (LOD2_UPDATE)

The state tensors live in the device buffer as part of the cache, so there is no
CPU to fall back to.  The lesson is the boring one: I read the abort's backtrace
line, inferred a mechanism from the source, and stated it.  The actual message
was two lines up.

The fix is entirely in the read path, because the shared-memory tiles have been
f32 since 4f.1 and do not care where a value came from: one code in
lod2_type_code and one case in lod2_get.  The CPU side already handled any type
with a to_float trait.

Verified, 2B, -fa on, refinement all-open vs dense and default vs dense, both
IDENTICAL for -ctk/-ctv f16 and q8_0, oracle 36/36.  27B/32k/1 GPU prefill is
unaffected: f16 1969.8, q8_0 1984.4 t/s.

### Also fixed here: an empty state at decode

Found while testing the above.  A prompt shorter than the front window creates no
slots, so a single-token decode fell through the s_len > 0 guard into the
many-query path and hit GGML_ASSERT(lgt != nullptr) - there is no routing GEMM
when there is nothing to route to.  Every consumer of lgt is bounded by s_len, so
the assert was simply wrong; it is now `lgt != nullptr || s_len == 0`.  Short
prompts with generation used to abort.

## 4j. Decode optimisation, and multi-stream (-np) decode

Baseline for everything below: 27B/Q4_K_M, 32k prompt, generation.  Per-token
GPU kernel time, taken as the difference between an `-n 65` and an `-n 1` run so
that prefill cancels out.  A trivial kernel dispatch costs 4.8 us in this setup
(`GGML_CUDA_DISABLE_GRAPHS=1`, measured on `lod2_k_iota`), so subtract that from
any per-call figure before calling it work.

Start: LoD2 decode kernels 3.65 ms/token against dense attention's 1.12.
End:   2.50 ms/token.  Generation on 8 GPUs 59.3 -> 63.7 t/s (dense 75.6).

### 4j.1 The routed branch was 74% of lod2_k_part, and parallelism was why

`LLAMA_LOD2_DBR` is a branch mask on lod2_k_part (1 coarse, 2 local, 4 routed);
it produces wrong output on purpose and exists to attribute cost.  It gave
coarse 7.0, local 5.4, routed 38.3, fold 3.0 ms over 528 calls - and within the
routed branch, capping the page scan and the exact columns separately gave page
scan 5.0, exact columns 33.1.

The columns were not latency bound.  Prefetching eight of them into registers
first (a scheduling change, arithmetic untouched) bought 6% and spilled the
kernel to scratch.  The cause was parallelism: `for (r = sp; r < LOD2_MAX_R;
r += NS)` gave route r to split r, so 8 of ~200 splits carried the whole branch
and 192 waves did all of it on a 304-CU device.

A split cannot know which columns it would read without the page argmax, and
that argmax is one answer per route.  So it moved out: lod2_k_route computes it
once per (unit, route), and the routed branch became one work item per residual
and per exact column, spread over R*(ps+1) splits instead of R.  lod2_k_part
52.1 -> 18.3 ms.  The merge is by logsumexp, which is associative, so which
split an item lands in does not change what the merge computes.

### 4j.2 Three kernels that did not respond to load batching

topk, reduce and logits_kv were each a loop whose loads looked serialised behind
the previous iteration's dependent work.  Batching four at a time in all three
changed nothing (2.09 -> 2.08 ms/token total).  That is the third time on this
port that a latency hypothesis has been wrong; the reductions are the pacing
item, not the loads.  What they do respond to is shape:

  * `lod2_k_reduce` is exactly linear in the partial count: 8 / 16 / 32 / 51 /
    101 partials cost 5.9 / 10.3 / 18.1 / 28.4 / 55.3 us.  `lod2_k_part` is flat
    in the split count above ~128.  The two are decoupled by folding wider
    rather than splitting less - a block of lod2_k_part now folds `pw` splits
    instead of a fixed 4, keeping its wave count and handing the reduce a
    quarter of the partials.  Swept together: (pw, NS) 4/128 115.8, 8/128 111.9,
    16/128 113.9, 8/202 122.5, 16/202 132.3 us per layer.  Defaults are pw = 8
    and NS aimed at ~10 warps per CU.
  * `lod2_k_topk` is 14.0 us of warp argmax merge, 7.8 of logit scan, and it
    grew by 20.7 when the page scan was briefly folded into it - not from the
    scan's work (0.7 us; making it scan one route instead of eight changed
    nothing) but from having it in a kernel with one block per unit, i.e. 24 of
    304 CUs busy.  On its own 192-block grid it costs 17.3 us including its
    dispatch.

### 4j.3 lod2_k_sim shares its scan, and is paced by the same reduction

One block per absorbed column re-read the whole slot table, so the same 3 MB per
head came back once per column - 256 of them at generation, 1280 at prefill.
`lod2_k_sim<MT>` stages MT columns and shares the scan; each (column, slot) dot
keeps its own lane-strided accumulation and warp reduction, and the block merge
is a max by (score, lowest slot) either way, so the result is unchanged.

A 4x traffic cut bought 1.4x: (MT, block) 1/256 614.8, 1/512 509.8, 4/256 550.6,
4/512 438.8, 8/512 483.0, 16/512 678.6 ms.  Same conclusion as 4j.2 - the
per-(column, slot) warp reduction paces it, and past four columns the register
file gives the sharing back.  Defaults MT = 4, block = 512.

The disassembly says the rest plainly: `lod2_k_sim<4>`'s inner loop is 24
`ds_bpermute_b32` (four warp sums of six levels) against 16 `ds_read_b32` and 16
FMAs.  `__shfl_xor_sync` at width 64 lowers to the LDS crossbar.  Removing the
reduction rather than speeding it up means giving a lane a whole dot product,
which changes the summation order - and the refinement chain's value is that it
is bit-stable, so that trade was not taken here.

### 4j.4 Where decode stands

Per token, 16 LoD2 layers: sim 0.63, part 0.62, topk 0.36, logits_kv 0.34,
reduce 0.17, route 0.14, merge 0.13, pages 0.08 = 2.50 ms, against dense
attention's 1.12.  Five dispatches per layer are ~0.38 ms/token of that with
graphs disabled.

### 4j.5 -np: LoD2 decode now runs for several sequences at once

Until now `build_attn_inp_lod2` refused any ubatch that was not single-stream,
so with `-np > 1` every decode ubatch - `llama_batch_allocr::split_equal`
concatenates the sequence sets - fell back to dense.  LoD2 did nothing during
generation at any np.  `LLAMA_LOD2_DBR=0` on the decode branches is what shows
this: it must change the output, and it did not.

split_equal makes a ubatch a run of equal-length groups, one sequence each,
appended in ascending sequence id.  So the ubatch is issued as one op chain per
group over that group's slice of the queries, with that group's stream, state,
schedule and archive, and the results concatenated.  No kernel, no op signature,
no CPU reference and no oracle changed, which is why the whole verification
chain still applies.  `set_input` asserts what the construction assumes: every
group is one sequence, positions consecutive from its head.

Two things had to be fixed to get there:

  * the context `graph_reserve` builds carries a dummy slot info of one cell per
    stream whatever the reserve ubatch is.  Reading the live coverage against
    its p0 of zero aborts on `l0 <= p0`.  That graph is only measured, so there
    the tokens are divided evenly and the state is taken as empty - which is
    also the widest plan, which is what a reservation wants.
  * `can_reuse` now refuses multi-group graphs outright.

Verified on the 2B model with `llama-parallel`, with a geometry small enough
(`chunk 8, local 16`) that short prompts still build a state, so the decode path
is the LoD2 one:

  np=1  refinement all-open vs dense IDENTICAL, DBR changes output
  np=2  refinement all-open vs dense IDENTICAL, DBR changes output
  np=4  refinement all-open vs dense DIFFERS

At three or more concurrent sequences the harness itself is not stable to
rounding: dense against dense with only the ubatch size changed (512 vs 384)
also DIFFERS, and LoD2 against dense is IDENTICAL at seed 7 and DIFFERS at 1234
and 99.  LoD2's exact path and dense flash attention agree numerically, not
bitwise, so a text-level verdict is not available there; what is available is
np=1 and np=2 exact, the structural assert on every ubatch, and the fact that
the same comparison passes for some seeds.

A note on measuring this: `grep "^Client"` matches nothing in llama-parallel's
output, because every line carries a timestamp and colour codes.  Two empty
files compare equal, so an entire round of "IDENTICAL" results here meant
nothing.  Compare the `Response:` lines with the escapes stripped, and print the
count so an empty comparison is visible.

## 4k. Command-line flags

LoD2 is configured from the command line the same way LoD is: each flag sets the
environment variable the engine already reads, so there is one place that
interprets a parameter and nothing downstream changed.  Available to every tool,
llama-server included.

  --lod2                     enable
  --lod2-growth F            slots = F*sqrt(N)          (16)
  --lod2-state-min N         floor on slots             (256)
  --lod2-local N             exact window at generation (512, multiple of chunk)
  --lod2-chunk N             absorb step / window grain (256)
  --lod2-routes N            slots opened per query     (8)
  --lod2-routes-prefill N    same, prompt processing    (3)
  --lod2-page N              tokens per page in a slot  (16)
  --lod2-pages-per-slot N    page table per slot        (128)
  --lod2-sink N              never-routed leading slots (1)
  --lod2-update N            columns per prefill update (5*chunk)
  --lod2-layers SPEC         "il,il,..." (default: all)
  --lod2-tiled MASK          tiled prefill kernels: 1 local, 2 coarse, 3 both (3)

The kernel tuning and debug knobs (LLAMA_LOD2_NS, PWPB, SIMMT, SIMB, RM, CT,
KVH, DBR, REUSE, ...) stay environment-only on purpose - they are for bisecting
this port, not for running it.

`--lod2-tiled` defaults to 3 as of 2026-08-14.  The tiled branches were opt-in
while the per-query kernel was the only verified path; they have since held
identical output through every refinement and oracle run, and they roughly
double prompt throughput, so the default follows the evidence.  0 still selects
the per-query kernel, which remains the reference implementation, and anything
that does not fit (shared memory, head sizes, an empty state) falls back to it
on its own.

Verified: `--lod2 --lod2-tiled 3` on its own raises 16 distinct LoD2 kernels and
1212 calls where a dense run raises none; flags and the equivalent environment
produce identical output; all-open through the flags still equals dense.

Two things to know when checking a run:

  * the confirmation line needs `-lv 4`.  At this fork's default verbosity of 3
    no llama_context INFO line is printed at all, LoD2's included, so its
    absence says nothing.  At `-lv 4`:
    `llama_context: LoD2 attention enabled: state = 24*sqrt(N) (min 256), ...`
  * do not try to confirm the flag by comparing output against dense.  At the
    sizes used here LoD2's default output *is* dense's output - that is the
    refinement property working - so an "IDENTICAL" tells you nothing about
    whether the flag took.  Count the kernels, or read the line above.

## 5. Standing constraints

no commits or pushes without explicit approval; ASCII only in files; all
cluster execution through `./simple-run.sh`; no model-specific code.
