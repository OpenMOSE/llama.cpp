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
llama-cli   -m model.gguf --lod [--lod-top-pages 32] [--lod-top-pages-decode 32] \
            [--lod-prefill gather|mask] [--lod-page-size 64] [--lod-sel layer|head] ...
llama-server -m model.gguf --lod ...
```

The flags work in every tool that uses common args (cli, server, bench, perplexity).
Underlying env vars (usable directly): `LLAMA_LOD=1`, `LLAMA_LOD_TOP_PAGES`,
`LLAMA_LOD_TOP_PAGES_DECODE`, `LLAMA_LOD_PREFILL`, `LLAMA_LOD_PAGE_SIZE`, `LLAMA_LOD_SEL`,
`LLAMA_LOD_FUSED=1` (fused decode kernel; `--lod` turns it on by default).

`--lod-top-pages` takes a per-layer spec: `DEF[,il=V|l0-l1=V ...]` where a value is
`PAGES`, `PAGES:REGIONS` or `dense` (that layer reads plain dense attention), e.g.
`"64:32,5=192:64,11=dense,20-30=128"`. `--lod-top-pages-decode` takes the same spec for
token generation and defaults to the prefill value; `dense` only takes effect from the
prefill spec, because the dense escape picks a different graph before the phase is known.

**Budget guidance (measured, see below): raise `--lod-top-pages`, leave the decode budget
alone.** Raising only the decode budget measurably *hurts* - it is not a quality knob.

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

Works with `-fa on` (recommended), with quantized KV (`-ctk q8_0 -ctv q8_0` runs the full
fast paths: the fused decode kernel dequantizes q8_0 blocks in-kernel, prefill uses the
static flash-attention graphs, and the mask-direct prefill reads q8_0 caches directly;
other quantized types fall back to dequantizing gathers), and with MTP speculative
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

Prefill has two modes, selected with `--lod-prefill`:

**`gather` (default)** - the selected leaves, the exact tail and the (1/ps scaled) page
sums are concatenated into one compact F16 KV set and evaluated with a single
`ggml_flash_attn_ext` call; log(n) bias and refinement silencing travel in the additive
mask, and the set is padded to `FATTN_KQ_STRIDE`. Selection is one top-k over
query/head-max-pooled page scores per layer per ubatch, so `-ub` acts as the "segment"
size of the LoD1 spec - larger ubatches see more recent context exactly (`-ub 2048`
recommended). **This is the only mode that delivers LoD's prefill speedup**: the KV set
size barely grows with depth, so prefill throughput is nearly depth-flat (measured
2816 t/s at 16k -> 2615 t/s at 82k, while dense falls 2519 -> 1308).

**`mask`** - flash-attention runs over `[page means | the raw cache span]` and the whole
selection lives in the additive mask, which lets each 32-query block choose its own pages.
That granularity is what makes multi-key retrieval scale with the budget (32k needle test:
0/3 -> 2/3 -> 3/3 at 16/64/128 pages, monotonic; `gather` is not monotonic - 2/3, 1/3, 3/3
at 32/64/128). **The cost is that the token tier is read in full, so mask prefill runs at
dense speed and decays with depth exactly like dense** (2461 -> 1239 over the same range;
dense 2519 -> 1308). Use it when retrieval reliability matters more than prompt speed.
Requires a cache type with resident page means (F16 or q8_0).

Decode (fused, default): one custom op `GGML_OP_LOD_ATTN` (CPU reference + HIP kernel,
`ggml/src/ggml-cuda/lod-attn.cu`) performs the whole read, including the page selection
(`sel == NULL` + `n_top`): a score kernel pools `dot(q, k_sums)` over queries and heads
(and folds the current token into the sums in the same launch), a shared-memory bitonic
sort picks the top pages, a split-K grid walks
[unselected summaries | selected leaves | tail] with one online softmax, and a merge
kernel combines the split partials - 4 kernels, no selection subgraph in the ggml graph
at all. Geometry (`n_past`, complete pages) travels through a tiny I32 input, so the
decode graph shape is fully static, reused every token, and replayed as one HIP graph;
selection ranks the *runtime* page count, so it never lags the static graph. F32, F16 and
q8_0 caches are all read natively by the kernel; other quantized types fall back to the
composed-op path for decode (correct, slower).

Decode selection is already per-query (one token = one query), so the decode budget only
buys *coverage*, and measurement says extra coverage does not help: on the coverage
benchmark (12 requirements scattered through 24k, all of them needed in one generation)
`pf32/dec32` scores 11/12 while `pf32/dec128` drops to **7/12**, losing a contiguous block
of mid-document requirements - reproduced identically three times. Raising the prefill
budget instead gives 12/12 and then the decode budget is irrelevant. The phase split is
therefore useful for *saving* decode work, not for spending more of it.

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

Prefill throughput vs depth (cold prompt, 1 GPU) - this is the property LoD exists for:

| mode | pp16384 | pp81920 |
| --- | --- | --- |
| LoD `gather` | 2816 t/s | **2615 t/s** (-7%) |
| LoD `mask` | 2461 t/s | 1239 t/s (-50%) |
| dense | 2519 t/s | 1308 t/s (-48%) |

Multi-GPU (`-sm layer`): prefill pipelines across devices. Three things must hold or the
pipeline collapses into a sequential layer chain (2 GPUs then run *slower* than 1):

1. the LoD graph must not be reused under pipeline parallelism - reuse makes
   `llama_context` synchronise the scheduler before filling inputs, stalling every device
   once per ubatch. `can_reuse` refuses reuse whenever `pipeline_parallel` is set.
2. the graph *shape* must stay stable across ubatches - a changed shape forces a
   scheduler reallocation, which is also a device-wide sync. The page tier is therefore
   padded to a stepped capacity (256 pages) whenever the budget is saturated, not sized
   by the live page count, and the validity vector silences the padding exactly.
3. that capacity must not be sized by `-c` either: the page-score tensor is
   `[P_cap, n_tokens, n_head_q]`, so padding to the whole context makes every ubatch pay
   for capacity it will never use (at `-c 262144` that is 4096 pages ~ 1 GB per layer).

Measured pp16384 @ depth 16k: LoD 2682 t/s on 1 GPU -> 4183 t/s on 2 GPUs (1.56x; dense
1702 -> 2977). Through `llama-server` (11k prompt, `-np 1`): 1717 -> 2237 t/s (1.30x;
dense 1827 -> 2159 = 1.18x). TG does not scale with a layer split for dense or LoD.

If a multi-GPU prefill shows no speedup, check VRAM headroom first: at long contexts the
non-SWA KV dominates (gemma4 at `-c 262144` is 20 GB of KV alone, plus ~10 GB of weights
per card), and once the cards are ~97% full the allocator thrashes and every reallocation
synchronises the devices. `-ctk q8_0 -ctv q8_0` halves the KV and is fully supported.

## Code map

- `src/llama-kv-cache.{h,cpp}` - page-sum tensors (`cache_{k,v}_page_l*`), LoD views
  (ranges, page rows, token rows, sums), clear/truncate bookkeeping, `can_shift = false`
- `src/llama-graph.{h,cpp}` - `llm_graph_input_attn_lod` (+`can_reuse`), `build_attn_inp_lod`,
  the LoD `build_attn` overload (page-sum updates, selection, FA-assembly / fused / composed reads)
- `ggml` - `GGML_OP_LOD_ATTN`: `ggml/src/ggml-cpu/ops.cpp` (reference),
  `ggml/src/ggml-cuda/lod-attn.{cu,cuh}` (split-K HIP kernel)
- `src/models/gemma4.cpp`, `src/models/qwen35.cpp` - per-model wiring
- `tests/test-lod-attention.cpp` - graph-level invariants (full expansion == dense at
  ~1e-14, quantized leaf gathers) on CPU and GPU backends, plus every accelerator result
  checked against the CPU implementation of the same op (the in-op and explicit-selection
  graphs share the read kernel, so their agreement alone proves only the selection stage)
- `lod-needle-mk3.sh`, `lod-coverage.sh` - retrieval and coverage benchmarks (see above).
  Both call `llama-cli` directly, so on a cluster they must be wrapped by the dispatcher
- `tools/lod-search/` - `llama-lod-search`, the per-layer budget profiler/allocator; the
  profile branch it consumes lives in `build_attn` behind `LLAMA_LOD_PROFILE`

Correctness anchors: the composed path with `-fa off -ctk f32 -ctv f32` is bit-identical
to dense; the fused op matches the composed path greedily; every change is regression-tested
against these.

## Choosing budgets and modes

Two benchmarks in the repo drive these numbers, and they measure different things:

- `lod-needle-mk3.sh <model> <6k|32k|100k> <n_ctx>` - three keys at 10/50/90% depth, one
  asked per run. Short answers, so it mostly measures whether the *first* generated token
  is right, which comes from the prefill logits.
- `lod-coverage.sh <model> <ctx_tokens> <n_ctx>` - twelve requirements scattered through
  the context, **all** of which must appear in one long generation. This is the one that
  reflects long-form work (writing code against a spec), and it prints which requirements
  were missed.

Measured on `lod-coverage.sh` at 24k (dense scores 12/12):

| prefill / decode budget | gather | mask |
| --- | --- | --- |
| 32 / 32 | 11/12 | 12/12 |
| 32 / 128 | **7/12** | 10/12 |
| 128 / 32 | 12/12 | 12/12 |
| 128 / 128 | 12/12 | 12/12 |

Practical reading: **spend the budget on prefill.** `--lod-top-pages 128` alone reaches
dense coverage; adding decode budget on top changes nothing, and adding it *without* the
prefill budget makes things clearly worse. `--lod-top-pages 128 --lod-top-pages-decode 32`
keeps the prefill cost (-17% pp) without the decode cost (top128 makes the decode kernel
1.6-2.0x slower, which is roughly -9% on end-to-end token generation).

## Per-layer budgets: `llama-lod-search`

Layers do not need the same budget: some full-attention layers carry the long-range
retrieval, others barely notice being starved. `llama-lod-search` finds that split and
writes it to a JSON file that `llama-cli` / `llama-server` load with `--lod-config`.

The tool has two very different modes, and it matters which one you trust:

| mode | what it measures | cost | trust |
| --- | --- | --- | --- |
| `--target ...` | attention *mass capture*, from one dense profiling pass | ~1 model run | a proxy - see the warning below |
| `--probes ... --search ...` | **retrieval log-probability**, by running the model at every candidate budget | 1 run per candidate | the objective you actually care about |

**Warning about the proxy.** The capture-based allocation is cheap because the deciding
quantity (how much dense attention mass lands in the pages the selector would pick) comes
out of a single forward pass. When it was validated against generation quality it did *not*
predict it: over repeated draws of `lod-coverage.sh` the capture-optimal split averaged
9.33/12 against 10.50/12 for a flat uniform budget at the same total. Use `--target` to
*understand* a model (the per-layer capture curves are a genuine diagnostic), not to pick a
configuration. Everything below uses the measured objective instead.

### The objective: retrieval log-probability

A pass/fail retrieval score is unusable as a search objective. `lod-coverage.sh` moves by
+-2 of 12 under nothing more than floating-point-level perturbation, so a search reading it
optimises noise. The probe objective measures the same ability continuously: for each
planted fact it teacher-forces `"\n<key> = "` after the document and reads the
log-probability of the correct value. No generation, no sampling, fully deterministic.

Calibration on gemma4-31B, 24k document, 12 probes, `--lod-prefill gather`:

| budget | retrieval logprob | probability of the right value |
| --- | --- | --- |
| top 4 | -3.9408 | 1.9% |
| top 8 | -3.4880 | 3.1% |
| top 16 | -2.3602 | 9.4% |
| top 32 | -1.6296 | 19.6% |
| top 64 | -0.9934 | 37.0% |
| top 128 | -0.0002 | 100.0% |
| saturating (= dense) | -0.0000 | 100.0% |

Monotonic over ~4 nats, and it says something the pass/fail score hides: at top 32 the model
answers most requirements, but it only puts ~20% of its probability mass on the planted
value. That headroom is what a per-layer search has to spend.

### Searching on the benchmark itself: `--niah`

Every cheap objective tried here has picked an allocation that then lost on the real
benchmark (the two sections after the workflow are the evidence). So the search can also run
the benchmark itself: `--niah file=value,...` prefills each needle document, generates
greedily exactly as `--temp 0` does, and counts how many documents contain the planted
value. Ties inside a tier are broken by the log-probability of the wanted string, so
candidates that all pass (or all fail) are still ordered.

```sh
# build needle documents from the coverage asset - one question per planted key
python3 - <<'PYEOF'
import json, os
src = "/tmp/lod-cov-24000.txt"
probes = json.load(open(src + ".probes.json"))
body = open(src).read().split("END OF SPECIFICATION.")[0] + "END OF SPECIFICATION.\n"
os.makedirs("/tmp/niah", exist_ok=True)
spec = []
for i in [0, 2, 4, 6, 8, 11]:                       # spread through the document
    key = probes[i]["prefix"].strip().split(" =")[0]
    path = f"/tmp/niah/{key}.txt"
    open(path, "w").write(body + f"\nQuestion: what value must the setting `{key}` be set to?"
                                 f"\nAnswer: {key} must be set to")
    spec.append(f"{path}={probes[i]['answer']}")
open("/tmp/niah/spec.txt", "w").write(",".join(spec))
PYEOF

./build/bin/llama-lod-search -m model.gguf -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --niah "$(cat /tmp/niah/spec.txt)" --search greedy \
    --eval-top 32 --range 8:128 --max-eval 30 -o model-lod.json
```

This is the honest objective and it is priced accordingly: one evaluation prefills every
needle document, so six documents at 26k tokens cost roughly a hundred times what a probe
evaluation costs. Use fewer documents, or a shorter context, if that is too slow - but do
not go back to optimising a proxy and skipping the check.

`--niah` with no `--search` just reports the score, which is the cheapest way to confirm the
objective moves with the budget before spending anything on a search. Measured on
gemma4-31B, six needle documents at 26.5k tokens:

| budget | documents answered | score |
| --- | --- | --- |
| top 8 | 0/6 | -0.0711 |
| top 32 | 2/6 | 1.9604 |
| top 128 | 5/6 | 4.9965 |

There is one scoring rule for every model: generate, and accept the document if the wanted
value appears anywhere in the output - exactly what `lod-needle-mk3.sh` does. Measured on
both wired families, same binary, same asset, same flags:

| budget | gemma4-31B | qwen3.6-27B |
| --- | --- | --- |
| top 8 | 0/6 | 2/6 |
| top 32 | 1/6 | 4/6 |
| top 128 | 4/6 | 6/6 |

Monotonic on both. Three implementation details were needed to get there, and all three are
generic - there is no per-model code anywhere in the tool:

- **Enable the fused decode path.** `--lod` sets `LLAMA_LOD_FUSED=1`; setting `LLAMA_LOD`
  alone leaves decode on the composed path, which nobody runs and which asserts (handover
  section 2d).
- **Never compute positions from the token count.** Under M-RoPE (qwen3.6) a token can
  advance the position by more than one - after 26750 tokens the cache is at 26752. Ask
  `llama_memory_seq_pos_max()` instead.
- **Never rewind and re-feed.** M-RoPE requires every batch to start strictly after the
  position already stored, so a rewind-then-replay is impossible. One forward pass per
  document, and read the pass/fail decision and the continuous tie-break out of that pass.

An exact "greedy must emit these tokens" rule was tried first and rejected: it works on
gemma4 and reports a flat, budget-insensitive score on qwen3.6 purely because that model
phrases its answer differently. That is a model-specific rule wearing the costume of a
metric, and it is exactly the trap this whole document keeps hitting.

Absolute counts still move by about one document under floating-point-level perturbation
(the same variance documented for `-b`/`-ub`). Treat the *shape* of the curve as the signal,
not the exact integer, and always finish with `lod-needle-mk3.sh` and `lod-coverage.sh` on
the config you intend to ship.

### Choosing the budget to search at (read this before running a search)

A search is run *at* a budget: `--eval-top N` fixes the total (N x number of LoD layers) and
the search only redistributes it. Which N you pick decides what the search can find, and the
wrong N makes the result look like a failure of the method when it is a failure of the
budget. Three regimes, measured on qwen3.6-27B at 24k (16 LoD layers, so `--eval-top 32` =
512 pages):

**Too low - peaks and floor compete.** At 512 pages the model cannot afford both a large
budget on the layers that carry retrieval and a survivable floor on the rest, so every shape
trades one benchmark for another:

| shape (512 pages) | needle | coverage |
| --- | --- | --- |
| uniform 32 | 4/6 | 12/12 |
| searched, floor 8, peaks 128 | 4/6 | 11/12 |
| floor 24, peaks 64-72 | 2/6 | 12/12 |

Nothing dominates. A search here measures the competition, not the allocation.

**Above that, on the 12-document scale, shaping has not beaten uniform at any budget
measured.** The full qwen3.6 ladder, needle scored on all twelve documents:

| configuration | pages | needle /12 |
| --- | --- | --- |
| uniform 32 | 512 | 8/12 |
| shaped, peaks 128 floor 24 | 696 | 8/12 |
| uniform 44 (equal cost) | 704 | 7/12 |
| uniform 64 | 1024 | 10/12 |
| shaped, peaks 192 floor 48 | 1200 | 10/12 |
| shaped, peaks 256 floor 64 | 1600 | 11/12 |
| uniform 128 | 2048 | 12/12 |

Uniform reaches 10/12 for 1024 pages where shaping needs 1200, and 12/12 for 2048 where
shaping gets 11/12 at 1600. On the six-document set these same runs looked like a 32%
saving; they were not. Budget, not shape, is what moved quality on both models tested.

**How to use this.**

1. Find where uniform saturates on *your* model and document, with three cheap runs:
   `--eval-top 16`, `32`, `64` scored by `--niah`. On qwen3.6 that is 8/12 -> 10/12 at 64,
   and coverage is already 12/12 at 32.
2. Search at roughly the budget where uniform is still clearly improving, not where it has
   flattened. Searching below that point hits the competition regime above.
2b. Before believing any searched config, put it next to uniform at the SAME total pages
   AND next to the uniform that reaches the same score. On both models tested so far the
   second comparison is the one that kills it.
3. If the searched config loses `lod-coverage.sh` to uniform, that is the floor talking.
   Raise `--eval-top` (or hand-raise the floor and keep the peaks) and re-measure both.
4. Always compare against uniform at **equal total pages**, not against the uniform you
   started from. `pages = sum of the per-layer budgets`; the tool prints it and the JSON
   records it as `budget_total`.

**Worked example, mid budget (the qwen result above):**

```sh
F=/tmp/lod-cov-24000.txt
N=$(cat /tmp/niah-24000/spec.txt)
Q=model.gguf

# search at 32/layer, floor left low so the peaks can be found
./build/bin/llama-lod-search -m $Q -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --niah "$N" --search greedy --eval-top 32 --range 8:128 --max-eval 22 -o cfg.json

# coverage lost a point? raise the floor by hand, keeping the peaks, and compare both
python3 - <<'PYEOF'
import json
c = json.load(open('cfg.json'))
hot = {k: v['prefill'] for k, v in c['layers'].items() if v['prefill'] >= 64}
c['layers'] = {k: {'prefill': hot.get(k, 24)} for k in c['layers']}
c['budget_total'] = sum(v['prefill'] for v in c['layers'].values())
json.dump(c, open('cfg-floor24.json', 'w'), indent=2)
print('floor-raised total:', c['budget_total'], 'pages')
PYEOF

# equal-cost control: uniform at total/n_layers, rounded
bash lod-coverage.sh   $Q 24000 32768 --lod-config cfg-floor24.json
bash lod-needle-mk3.sh $Q 32k   32768 --lod-config cfg-floor24.json
bash lod-coverage.sh   $Q 24000 32768 --lod --lod-top-pages 44
```

**Worked example, high budget (matching dense-level retrieval):**

```sh
# start the search from the budget you actually want to beat, not from 32
./build/bin/llama-lod-search -m $Q -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --niah "$N" --search greedy --eval-top 96 --range 48:256 --max-eval 22 -o cfg-hi.json
```

At this level keep `--range`'s floor high (about half the baseline): the quiet layers are no
longer free, and a search that starves them will trade away exactly the last documents you
are trying to win.

**Cost.** One evaluation prefills every needle document. With the default 12-document asset
that is roughly 2 minutes per candidate on a 27-31B model at 24k, so a 22-evaluation search
is about 45 minutes. Halving the document count halves the time and doubles the noise (each
document is one discrete point) - do that only for a quick smoke test, not for a decision.

### Workflow

Everything below is copy-pasteable. Replace the model path; nothing else is model-specific.

**Step 0 - build the calibration asset.** `lod-coverage.sh` writes both the document and
the probe sidecar, and `COV_ASSET_ONLY=1` stops before the generation benchmark:

```sh
# writes $LOD_ASSET_DIR/lod-cov-24000.txt and lod-cov-24000.txt.probes.json
COV_ASSET_ONLY=1 LOD_ASSET_DIR=/tmp bash lod-coverage.sh anything 24000 32768
```

The asset is plain text with no chat template, so the *same* file works for gemma, qwen and
anything else. Size it to the context you actually care about - a config calibrated at 24k
says nothing about 100k.

**Step 1 - check the objective responds on your model.** If the numbers do not move with the
budget, nothing downstream is meaningful:

```sh
F=/tmp/lod-cov-24000.txt
for T in 8 32 128 1000000; do
  ./build/bin/llama-lod-search -m model.gguf -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
      --probes $F.probes.json --eval-top $T | grep "retrieval logprob"
done
```

**Step 2 - per-layer sensitivity.** Hold every layer at the baseline budget, starve one
layer to `--range`'s minimum, and see what it costs. This is the direct measurement of
"which layers need budget", at one model run per LoD layer:

```sh
./build/bin/llama-lod-search -m model.gguf -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --probes $F.probes.json --search sens --eval-top 32 --range 8:256
```

**Step 3 - search under a fixed budget.** `--search greedy` keeps the *total* page count
equal to the uniform baseline (`--eval-top` x number of LoD layers) and redistributes it:
first by shaping the allocation along the measured sensitivity (several exponents), then by
single-step transfers from the cheapest layer to the dearest. Every candidate costs the same
as uniform, so an accepted move is a strict win at equal cost:

```sh
./build/bin/llama-lod-search -m model.gguf -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --probes $F.probes.json --search greedy --phase prefill \
    --eval-top 32 --range 8:128 --guard 256 --guard-tol 0.02 \
    --max-eval 45 -o model-lod.json
```

`--max-eval` is the stop condition. Each evaluation is one context build plus one full
prefill of the calibration document plus the probes - measured at 12 s for gemma4-31B at
24k on an MI325X, so a 45-evaluation search takes about nine minutes. The model is loaded
once and only the context is rebuilt per candidate, which is what makes that possible.

`--guard` is not optional in practice. It holds the last N document tokens out of the
prefill and scores them as ordinary next-token prediction, and any candidate that loses
more than `--guard-tol` nats there is rejected however good its retrieval is. Read the next
section for why.

**Step 3b - or trim instead of redistribute.** On gemma4 the redistribute-at-fixed-cost
direction did not pay off (next section), but the same measurement supports the opposite
move: start from a budget that is already known to be good and take pages away from the
layers that do not miss them. `--search trim` halves each layer's budget, least sensitive
first, and keeps every cut that leaves retrieval within `--tie` of where it started:

```sh
./build/bin/llama-lod-search -m model.gguf -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --probes $F.probes.json --search trim --phase prefill \
    --eval-top 128 --range 8:128 --guard 128 --tie 0.05 \
    --max-eval 60 -o model-lod.json
```

This answers "how cheap can this get without measurably losing retrieval", which is the
question a deployment actually asks. Validate it the same way as anything else.

**Step 4 - the decode budget, into the same file.** `--phase decode` feeds the probes one
token at a time so the graph takes the decode branch, pins prefill at a saturating budget so
only the decode budget moves, and merges its result into the existing JSON:

```sh
./build/bin/llama-lod-search -m model.gguf -f $F -c 32768 -ub 2048 -ngl 99 -fa on \
    --probes $F.probes.json --search greedy --phase decode \
    --eval-top 32 --range 8:128 --guard 256 --guard-tol 0.02 \
    --max-eval 30 -o model-lod.json
```

**Step 5 - validate before trusting it.** The search optimises retrieval log-probability on
one document; that is a proxy for what you care about, so confirm on the end-to-end tests -
and treat needle retrieval as the thing that must not regress:

```sh
bash lod-needle-mk3.sh model.gguf 32k 32768 --lod-config model-lod.json    # must not drop
bash lod-coverage.sh  model.gguf 24000 32768 --lod-config model-lod.json   # vs uniform
```

**Step 6 - use it.**

```sh
llama-server -m model.gguf --lod --lod-config model-lod.json -c 32768 -ngl 99 -fa on
llama-cli    -m model.gguf --lod --lod-config model-lod.json -c 32768 -ngl 99 -fa on
```

### The config file

`--lod-config` is translated into the same per-layer spec grammar as `--lod-top-pages`, so
there is exactly one code path that interprets budgets. Layer numbers are the model's
absolute layer indices, so a config belongs to one model:

```json
{
  "page_size": 64,
  "n_ctx_calibrated": 32768,
  "n_tokens_calibrated": 23994,
  "objective": "retrieval_logprob",
  "default":  { "prefill": 32, "decode": 32 },
  "layers": {
    "23": { "prefill": 96, "decode": 64 },
    "59": { "prefill": 64 }
  }
}
```

`default` applies to every LoD layer the file does not name. A layer entry may set
`prefill`, `decode` and `regions` independently; anything absent falls back to `default`,
then to the command line. Writing the file by hand is fine - the search is one way to
produce it, not the only one.

### What it found on gemma4-31B (24k document, gather prefill)

Per-layer sensitivity, all ten LoD layers at 32 pages except one starved to 8. Baseline
retrieval log-probability -1.6296:

| layer | logprob when starved | cost of starving |
| --- | --- | --- |
| 23 | -2.4347 | **+0.8051** |
| 59 | -1.9717 | +0.3421 |
| 41 | -1.9356 | +0.3060 |
| 5 | -1.9088 | +0.2792 |
| 35 | -1.8055 | +0.1759 |
| 29 | -1.8032 | +0.1736 |
| 11 | -1.7265 | +0.0969 |
| 47 | -1.7096 | +0.0800 |
| 53 | -1.7084 | +0.0788 |
| 17 | -1.4961 | -0.1335 |

So the layers are *not* equal - layer 23 costs eight times what the median layer costs, and
layer 17 is slightly *better* starved. That is the measurement the whole idea rests on, and
it holds.

### The needle objective disagrees with the proxy - and it is the one to believe

Running `--search sens` with `--niah` (six needle documents built from the same 24k
document, scored by real greedy generation) gives a completely different ranking from the
teacher-forced probe. Baseline is all ten LoD layers at 32 pages, which answers 2 of 6;
each row starves one layer to 8:

| layer | needle documents lost | rank on the needle test | rank on the probe |
| --- | --- | --- | --- |
| 11 | 2.02 | 1 | 7 |
| 29 | 2.01 | 2 | 6 |
| 23 | 2.00 | 3 | 1 |
| 41 | 1.01 | 4 | 3 |
| 35 | 1.00 | 5 | 5 |
| 5 | 0.02 | 6 | 4 |
| 17 | 0.02 | 7 | 10 |
| 59 | 0.01 | 8 | 2 |
| 47 | 0.00 | 9 | 8 |
| 53 | -0.00 | 10 | 9 |

Five layers carry the long-range retrieval and five can run at the floor - the premise that
motivated this whole tool. But the proxy put layer 11 seventh (it is first) and layer 59
second (it is eighth), so a budget allocated from the proxy spends on the wrong layers. If
you only take one thing from this document: **rank layers with `--niah`, not with
`--probes`.** The probe objective is still useful as a fast smoke test that the budget
matters at all; it is not a ranking.

Note also what the baseline says: at uniform 32 this model answers 2/6, and starving almost
any middle layer drops it to 0/6. That is a cliff, not a gradient - searching around a
budget with more headroom (top 64 or 96) gives the search somewhere to move.

### What the searched allocations actually deliver: nothing measurable

This is the section that matters, and it is a negative result. Read it before spending an
hour on a search.

The search does what it is asked: driven by the needle objective it concentrates gemma4's
320 pages onto layers 11/29/23 (104/104/56, everything else at the floor) and qwen3.6's
onto 11/19/31. On the six-document needle set those configurations looked like clear wins -
gemma 2/6 against uniform's 1/6, qwen 5/6 against 3/6 at equal cost. **Both wins disappeared
when the needle set was widened from 6 documents to 12.**

gemma4-31B, all rows at 320 pages except the references:

| configuration | pages | needle /12 | coverage | mk3 |
| --- | --- | --- | --- | --- |
| uniform 32 | 320 | 3/12 | 11/12 | 3/3 |
| searched (floor 8) | 320 | 3/12 | 12/12 | 3/3 |
| floor 24 variant | 320 | 2/12 | 12/12 | - |
| uniform 64 | 640 | 4/12 | 12/12 | - |
| uniform 128 | 1280 | 6/12 | 12/12 | 3/3 |

qwen3.6-27B:

| configuration | pages | needle /12 |
| --- | --- | --- |
| uniform 32 | 512 | 8/12 |
| shaped, peaks 128 floor 24 | 696 | 8/12 |
| uniform 44 (equal cost to the row above) | 704 | 7/12 |
| uniform 64 | 1024 | 10/12 |
| shaped, peaks 192 floor 48 | 1200 | 10/12 |
| shaped, peaks 256 floor 64 | 1600 | 11/12 |
| uniform 128 | 2048 | 12/12 |

Read the qwen table by pages: uniform 64 reaches 10/12 for 1024 pages while the shaped
config needs 1200 for the same score, and uniform 128 reaches 12/12 where the shaped 1600
gets 11/12. **Per page, uniform is at least as good at every point measured.** The single
row that favours shaping (696 pages 8/12 against 704 pages 7/12) is one document out of
twelve.

The per-layer differences are real and reproducible - starving qwen's layer 19 costs two
documents, starving layers 3/7/23 *gains* one - but on these two models that structure does
not convert into an allocation that beats spending the same pages evenly. If you want more
quality, the measurement says: raise the budget, do not reshape it.

**The methodological point is the durable one.** Every apparent win in this line of work
sat inside the resolution of whatever metric was being used at the time:

| metric | resolution | what it claimed | what survived |
| --- | --- | --- | --- |
| attention-mass capture | continuous | searched split wins | 9.33/12 vs 10.50/12 over repeats - lost |
| retrieval log-probability | continuous | concentrate everything on one layer | coverage 2/12 - lost |
| needle, 6 documents | 6 points | gemma +1, qwen +2 at equal cost | ties at 12 documents - lost |
| needle, 12 documents | 12 points | no gain | (current state) |

A difference of one point on a six-point scale is not a result. The needle asset now
defaults to **all twelve** planted keys for exactly this reason; halving it to save runtime
also halves the resolution you are making decisions with.

### gather quality is NOT monotone in the budget (and mask is)

The single most useful measurement in this whole line of work, found while trying to explain
why coverage kept moving. gemma4-31B, 24k document, coverage scored at three nearby budgets,
each run twice (temp 0, byte-identical repeats):

| budget | pages | gather | mask |
| --- | --- | --- | --- |
| top 40 | 400 | 12/12 | 12/12 |
| top 48 | 480 | **7/12** | 12/12 |
| top 56 | 560 | **7/12** | 12/12 |
| top 64 | 640 | 12/12 | - |

Raising the budget from 400 to 480 pages *loses five requirements*, and raising it further to
640 gets them back. The generation is not truncated - it writes seven settings and simply
skips requirements 6 to 10, which sit in a contiguous stretch of the document:

```
alpha_timeout = 4172     <- present
...
echo_depth = 9318        <- present
kilo_budget = 2761       <- present   (foxtrot..juliet, the middle five, are gone)
lima_ceiling = 5528      <- present
```

This is the shared-page-set weakness of `gather` made concrete: one page set serves the
whole ubatch, so a particular budget can select a set that covers no page in one region of
the document, and every query in that ubatch loses it together. `mask`, which selects per
32-query block, is stable at every budget tested.

Consequences, all of which cost time in this work before the cause was known:

- **Do not tune a gather budget by taking one measurement per budget.** Quality is not
  monotone, so a single point can be an outlier in either direction. Sweep, or use mask.
- **A 1-of-12 coverage difference between two configurations means nothing** when adjacent
  budgets differ by 5. Several comparisons earlier in this document were read that way and
  should not have been.
- If long-form quality matters more than prefill speed, this is a concrete argument for
  `--lod-prefill mask` beyond the depth-scaling trade already documented.

### Sensitivity is real even where the allocation does not pay

Worth keeping separately from the negative result above, because it is reproducible and it
is the part a future attempt should build on. qwen3.6-27B, 16 LoD layers, one layer starved
to 8 pages at a time against a uniform 32 baseline:

| layer | needle documents lost |
| --- | --- |
| 19 | +2 |
| 31, 11 | +1 each |
| 15, 27, 35, 39, 43, 47, 51, 55, 59, 63 | ~0 |
| 3, 7, 23 | **-1 each (starving them GAINS a document)** |

gemma4-31B is similar in shape: layers 11/29/23 carry the retrieval, 5/17/47/53 cost nothing
to starve. The signal is not noise - it repeats to four decimals on the continuous
tie-break. What has not worked, three different ways, is turning it into pages.

### The trap: do not optimise retrieval alone

Run without `--guard`, the search follows that ranking to its logical end: it gives layer 23
248 of the 320 pages, drops every other layer to 8, and reports retrieval
log-probability -0.0011 - indistinguishable from dense - at exactly the uniform budget.
It is a 1.6-nat improvement on the objective, and it is a disaster:

| configuration | total pages | retrieval logprob | NIAH mk3 32k | coverage 24k |
| --- | --- | --- | --- | --- |
| uniform top32 | 320 | -1.6296 | 2/3 | 11/12 |
| searched, no guard | 320 | **-0.0011** | **1/3** | **2/12** |
| uniform top128 | 1280 | -0.0002 | 3/3 | 12/12 |

Teacher-forced retrieval only asks the model to copy a value it can see; one layer with
near-dense reach satisfies it completely. Generating four hundred coherent tokens against
the same document needs *every* layer to still read enough context, and at 8 pages
(512 tokens of a 24k document) they cannot. The objective was not wrong, it was
incomplete, and an optimiser will find that hole every time.

`--guard` was added against exactly this: the last N tokens of the document are scored out
of the same prefill, as a KL divergence against the dense read of the same positions.
(Plain next-token NLL was tried first and is not enough - a starved model gets *more*
confident on repetitive text, so its NLL improves while its distribution is wrong.)

But be clear about how far that gets you, because the measurement is unambiguous. Capping
the range at `8:128` and running the same search with the KL guard gives layers 23 and 59
128 pages and everything else 8, and on that allocation **all three cheap metrics improve
at once** - retrieval -1.6296 to -1.2260, KL-vs-dense 4.16 to 2.95 - while generation gets
worse:

| configuration | retrieval | KL vs dense | NIAH mk3 32k | coverage 24k |
| --- | --- | --- | --- | --- |
| uniform top32 | -1.6296 | 4.16 | 2/3 | 11/12 |
| searched, no guard | -0.0011 | - | 1/3 | 2/12 |
| searched, KL guard | -1.2260 | 2.95 | 1/3 | 9/12 |

So on gemma4-31B, **redistributing a fixed page budget across layers has not beaten a flat
budget end-to-end, under any objective tried.** The per-layer differences are real (the
sensitivity table above is reproducible to four decimals), they simply do not convert into
a better allocation: copying a value needs one layer with reach, and generating four
hundred coherent tokens needs every layer to still read context. Take the sensitivity scan
as a diagnostic, use `--search trim` if you want to spend the measurement on cost (below),
and keep the budget uniform otherwise.

The general lesson, which this cost three rounds to learn: **a cheap objective must be
validated against the expensive one before it is trusted, on a task the proxy does not
resemble.** Capture failed that test. Retrieval log-probability failed it differently.
Retrieval plus a KL guard failed it again, more subtly. All three failures were visible
only because `lod-coverage.sh` and `lod-needle-mk3.sh` were run against the result.

### Model notes

- **gemma4** - LoD is wired into the full-attention layers only, which on the 31B are every
  sixth layer (5, 11, ... 59; ten layers, head_dim 256). A "budget of 32" therefore means
  320 pages in total, not 32 per model layer.
- **qwen3.5 / qwen3.6** - same picture; the full-attention layers are all head_dim 256. The
  layer *count* differs, so run the sensitivity scan rather than porting gemma's split.
- Page size is fixed at 64 tokens (`ps`), so a page budget of `k` reads `64*k` tokens
  exactly per layer, plus the summaries and the exact tail.
- The tool discovers the LoD layers itself by running one profiled pass over the
  calibration document, so `--search` never wastes evaluations on layers that would ignore
  their budget.

## Known gaps / next steps

- **`llama-server` multi-turn generation can collapse into gibberish, intermittently.**
  Not root-caused. `LLAMA_LOD_CHECK=1` (plus `-v`, the library log is suppressed
  otherwise) recomputes every page sum from the cache cells at each prefill and reports
  the first layer/page that disagrees with the incrementally maintained value; that
  report will name whichever cache edit desynchronised the bookkeeping. `LLAMA_LOD_CHECK=2`
  first corrupts one page on purpose so you can confirm the detector fires.
- selection precision still degrades past ~64k: at 100k the needle test needs
  `top_pages` 1024 with `gather`; the region tier (selection-only, max-pooled over 8-page
  regions, active in `mask` mode) did not rescue it, so this is score SNR at ~2000
  candidates rather than a pruning problem. Pre-RoPE selection sums are the untested
  hypothesis.
- `gather` retrieval quality is not monotonic in the budget (32k: 2/3, 1/3, 3/3 at
  32/64/128 pages). `mask` is monotonic but gives up the prefill speedup entirely.
- fused kernel reads F32/F16/q8_0 caches natively; other quantized types use the
  composed fallback
- per-query selection as in the LoD1 spec is future work; `mask` gives per-32-query-block
  selection, `--lod-sel head` gives per-KV-head sets at x n_head_kv gather cost
