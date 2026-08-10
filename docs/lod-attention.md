# LoD sparse attention (experimental)

LoD (level-of-detail) is a training-free sparse read for the full-attention layers of a model.
The KV cache is kept unchanged, but reads go through a two-tier index:

- **page** - a run of `page_size` (default 64) contiguous tokens. Each page keeps one running
  **sum** of its keys and values (F32, accumulated from the raw pre-quantization K/V, so the
  summaries are never polluted by cache quantization).
- **leaf** - the raw cache entries of a page.

A query scores every complete page, expands the best `top_pages` into leaves, and attends
exactly there. Every *unselected* page still contributes its count-weighted mean plus a
`log(page_size)` size term, so the softmax denominator is always complete - this is
*refinement*, not plain top-k pruning: expanding every page reproduces dense attention
exactly. The most recent tokens (the partial page plus the current ubatch) are always read
exactly ("tail"). Reference: `/home/mose/Projects/RWKVInside4/docs/lod1_spec.md` (LoD1;
this port implements the all-pages branch, i.e. no region tier, valid to ~64k of context
at the default settings). The full algorithm-level specification of this port - written
for independent (e.g. PyTorch) re-implementation - is in `lod-attention-spec.md`;
the PyTorch reproduction package (naive reference + the production kernels wrapped
as a torch extension + parity test) is in `lod-pytorch/`, and the current work state
/ open investigations are in `lod-attention-handover.md`.

## Usage

```sh
llama-cli   -m model.gguf --lod [--lod-top-pages 32] [--lod-page-size 64] [--lod-sel layer|head] ...
llama-server -m model.gguf --lod ...
```

The flags work in every tool that uses common args (cli, server, bench, perplexity).
Underlying env vars (usable directly): `LLAMA_LOD=1`, `LLAMA_LOD_TOP_PAGES`,
`LLAMA_LOD_PAGE_SIZE`, `LLAMA_LOD_SEL`, `LLAMA_LOD_FUSED=1` (fused decode kernel; `--lod`
turns it on by default).

Supported models: **gemma4** and **step35** (LoD on the non-SWA layers; SWA layers
untouched), **qwen35 / qwen3.6** and **qwen35moe** (LoD on the full-attention layers;
the GDN linear layers untouched), and **hy-v3** (every layer is full attention - the
model class where LoD decode also wins at depth). Any other model runs unmodified even
with `--lod` set.

`llama-server` note: LoD needs per-sequence KV streams. With `-np` unset the server
would auto-enable the unified cache, which silently falls back to the dense read on
every ubatch - `--lod` now keeps `kv_unified = false` in that auto path, and an
explicit `--kv-unified` logs a warning. Pass `-np N` explicitly when in doubt.

Always pass `-ub 2048` (the server default is 512): LoD prefill pays a fixed
per-ubatch cost for selection and KV assembly, and at `-ub 512` that cost dominates -
measured at depth 64k: LoD 708 t/s at `-ub 512` vs 2546 t/s at `-ub 2048` (dense at
`-ub 512`: 911 t/s).

Works with `-fa on` (recommended), with quantized KV (`-ctk q8_0 -ctv q8_0` etc. - leaves
and the exact tail are read through dequantizing gathers), and with MTP speculative
decoding (`--spec-type draft-mtp`).

### Restrictions

- parallel slots are supported (`llama-server -np 4` etc.): each stream keeps its own
  page sums. A ubatch that mixes tokens of several streams falls back to a dense read
  for that ubatch (correct - the cache always holds the dense contents; the sums catch
  up lazily afterwards)
- no context shift (the cache reports `can_shift = false`); suffix removal (prompt reuse,
  speculative rollback) is supported, mid-range removal forces a reprocess
- `--lod-page-size` must stay in sync between runs of the same process only (no persistence)

## How it reads

Prefill (`n_ubatch > 8`): the selected leaves, the exact tail and the (1/ps scaled) page
sums are concatenated into one compact F16 KV set and evaluated with a single
`ggml_flash_attn_ext` call; log(n) bias and refinement silencing travel in the additive
mask, and the set is padded to `FATTN_KQ_STRIDE`. Selection is one top-k over
query/head-max-pooled page scores per layer per ubatch, so `-ub` acts as the "segment"
size of the LoD1 spec - larger ubatches see more recent context exactly (`-ub 2048`
recommended; at that setting LoD prefill quality matches dense).

Decode (fused, default): one custom op `GGML_OP_LOD_ATTN` (CPU reference + HIP kernel,
`ggml/src/ggml-cuda/lod-attn.cu`) performs the whole read, including the page selection
(`sel == NULL` + `n_top`): a score kernel pools `dot(q, k_sums)` over queries and heads
(and folds the current token into the sums in the same launch), a shared-memory bitonic
sort picks the top pages, a split-K grid walks
[unselected summaries | selected leaves | tail] with one online softmax, and a merge
kernel combines the split partials - 4 kernels, no selection subgraph in the ggml graph
at all. Geometry (`n_past`, complete pages) travels through a tiny I32 input, so the
decode graph shape is fully static, reused every token, and replayed as one HIP graph;
selection ranks the *runtime* page count, so it never lags the static graph. Quantized
caches currently fall back to the composed-op path for decode (correct, ~13% slower -
same order as the dense q8_0 penalty).

Ablation switches for the decode read (same binary): `GGML_LOD_SPLIT=lds` selects an
LDS-staged split kernel (one block per KV head, K/V staged through shared memory once
per column); `GGML_LOD_SPLIT=mega` selects a single persistent mega-kernel (all four
stages in one launch, software grid barriers between phases; wave64, P<=2048).
Both measured slower than the default pipeline in every regime (incl. 100k-scale
geometry) and are kept for ablation. `LLAMA_LOD_FUSED=0` selects the composed-op
read; `LLAMA_LOD_SEL=head` selects per-KV-head page sets. The micro-harness
(`LOD_BENCH=1 test-lod-attention`, with `LOD_BENCH_TOP` / `LOD_BENCH_P`) measures the
fused op at arbitrary budget/page-count geometry.

## Measured results (MI325X, single GPU, `-ub 2048`, top_pages 32)

gemma-4-31B q4_0 (10/60 full-attention layers, D=512):

| metric | dense (fa on) | LoD |
| --- | --- | --- |
| pp16384 @ depth 16k | 1706 t/s | **2309 t/s** (+35%) |
| pp16384 @ depth 48k | 1045 t/s | **2292 t/s** (+119%, depth-flat) |
| tg512 @ depth 0 | 51.3 | 51.3 (100%) |
| tg512 @ depth 16k | 50.2 | 47.7 (95%) |
| tg512 @ depth 48k | 47.5 | 44.1 (93%) |
| perplexity (c=4096) | 91.92 | 91.50 |
| needle 6k ctx, top8 (8% far-context read) | ok | ok (7391) |

Qwen3.6-27B Q4_K_M (16/64 full-attention layers, D=256):

| metric | dense (fa on) | LoD |
| --- | --- | --- |
| pp16384 @ depth 48k | 2011 t/s | **2490 t/s** (+24%, depth-flat) |
| tg512 @ depth 48k | 44.2 | 43.9 (99%) |
| perplexity (c=4096) | 4.298 | 4.269 |

(absolute t/s vary a few percent run to run with the node's power state; the ratios
are from adjacent same-condition runs)

With `-ctk q8_0 -ctv q8_0` (gemma): prefill identical to the F16 LoD numbers, PPL parity,
full-attention KV footprint ~0.53x (q4_0: ~0.28x).

Multi-GPU (`-sm layer`): prefill pipelines across devices like dense. The mask and the
page-tier shapes are generated on-device at fixed capacity, so the prefill graph is
static; under pipeline parallelism the graph takes the rebuild path (reuse would force
a scheduler sync per ubatch), on a single GPU it is reused. Measured pp16384 @ depth
16k: LoD 2682 t/s on 1 GPU -> 4183 t/s on 2 GPUs (1.56x; dense scales 1702 -> 2977).
TG does not scale with layer split for dense or LoD (sequential layer chain).

## Code map

- `src/llama-kv-cache.{h,cpp}` - page-sum tensors (`cache_{k,v}_page_l*`), LoD views
  (ranges, page rows, token rows, sums), clear/truncate bookkeeping, `can_shift = false`
- `src/llama-graph.{h,cpp}` - `llm_graph_input_attn_lod` (+`can_reuse`), `build_attn_inp_lod`,
  the LoD `build_attn` overload (page-sum updates, selection, FA-assembly / fused / composed reads)
- `ggml` - `GGML_OP_LOD_ATTN`: `ggml/src/ggml-cpu/ops.cpp` (reference),
  `ggml/src/ggml-cuda/lod-attn.{cu,cuh}` (split-K HIP kernel)
- `src/models/gemma4.cpp`, `src/models/qwen35.cpp` - per-model wiring
- `tests/test-lod-attention.cpp` - graph-level invariants (full expansion == dense at
  ~1e-14, quantized leaf gathers) on CPU and GPU backends

Correctness anchors: the composed path with `-fa off -ctk f32 -ctv f32` is bit-identical
to dense; the fused op matches the composed path greedily; every change is regression-tested
against these.

## Known gaps / next steps

- **selection precision degrades past ~64k of context** (the documented boundary is
  real): on a synthetic 100k needle, dense answers correctly but LoD misses at
  `top_pages` 32 and 64 and recovers at 128 - with identical behavior from the
  composed (graph) and fused (in-op) selection, so this is the selection budget, not
  a kernel bug. Until the region tier lands, use `--lod-top-pages 128` (or more) for
  100k+ prompts; note the decode leaf walk grows with the budget (~46 t/s at top32 vs
  ~31 t/s at top128 at depth 100k)
- region tier (restores selection precision at low budgets for >64k contexts) not
  implemented - the highest-priority next step
- fused kernel is F16-only (quantized decode uses the composed fallback); q8_0 in-kernel
  dequant is designed but not implemented
- parallel sequences and multi-GPU input broadcast optimization not implemented
- selection is shared per layer (or per KV head with `--lod-sel head`, at x n_head_kv
  gather cost); per-query selection as in the LoD1 spec is future work
