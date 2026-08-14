# LoD attention - handover notes (as of 2026-08-11)

Companion documents:
- `lod-attention.md` - user-facing: usage, flags, measured results, code map
- `lod-attention-spec.md` - algorithm spec, sufficient for independent reproduction
- `lod-pytorch/` - PyTorch reproduction package: naive reference
  (`lod_attention_ref.py`, self-testing), the production kernels extracted
  verbatim + torch JIT binding (`lod_attn_kernels.cu`, `lod_attn_ext.py` with
  a CUDA-vs-reference parity test), and its own README/handover

Everything below is uncommitted work on branch `lod`.

## 1. What is DONE and validated

- Full port of the LoD1 all-pages branch: exact-denominator refinement read
  (summaries/leaves/tail, one softmax), bitwise dense at full expansion.
- Fused decode: 4 kernels (score+fold, bitonic top-k, split, merge), in-op
  selection over the runtime page count, HIP-graph replayed, wave64/wave32
  (RDNA3) via runtime tile dispatch.
- Static capacity-padded prefill graphs + on-device mask generation ->
  multi-GPU pipeline parallelism works: pp16384@d16k 2744 (1 GPU, dense 1702)
  -> 4246 (2) -> 6586 (7). THREE couplings, all of them found the hard way:
  (a) under pipeline parallelism the graph must take the REBUILD path (llama.cpp
  synchronizes reused graphs per ubatch) - the guard now covers every LoD graph,
  it used to sit inside the static-graph branch only, which meant real (never
  page-aligned) prompts lost pipelining entirely; (b) the graph SHAPE must be
  stable across ubatches, so the page tier is padded to a stepped capacity
  whenever the budget is saturated - sizing it by the live page count reshapes
  every ubatch and each reshape is a scheduler realloc = device-wide sync;
  (c) that capacity must not follow `-c` either (the score tensor is
  `[P_cap, n_tokens, n_head_q]`; at -c 262144 that is ~1 GB per layer of pure
  padding). See `can_reuse` and `lod_page_capacity` in llama-graph.{h,cpp}.
- Phase-split budgets: `--lod-top-pages-decode` (same per-layer grammar as
  `--lod-top-pages`, defaults to it). Measured: raising ONLY the decode budget
  hurts coverage (11/12 -> 7/12), raising the prefill budget reaches dense
  (12/12). The split is for saving decode work, not spending more.
- Prefill mode switch promoted to a flag: `--lod-prefill gather|mask`
  (default gather). Mask = per-32-query-block selection, monotonic retrieval
  quality, but its token tier is read in full so prefill runs at DENSE speed and
  decays with depth like dense (2461->1239 over 16k->82k while gather holds
  2816->2615). Gather is the only mode that delivers the prefill speedup.
- State machine under mutation: suffix truncate = page-floor rewind + physical
  row zeroing + lazy catch-up refold. This fixed real corruption after server
  prompt-cache reuse AND after MTP rollbacks (folds are read-modify-write; a
  bare watermark rewind double-counts).
- Multi-slot server (-np N, per-stream sums, dense fallback for mixed batches),
  MTP (gemma assistant gguf + qwen3.6 built-in). q8_0 KV runs the full fast paths
  (in-kernel dequant in the fused decode kernel, static-graph FA prefill, mask
  prefill on the raw q8_0 cache); other quantized types use the composed path.
- Wired models: gemma4, step35 (iSWA), qwen35, qwen35moe (hybrid), hy-v3
  (pure full attention). step35/hy-v3 compile-validated only (no ggufs here).
- Server traps fixed: auto kv_unified (would silently disable LoD; now kept off
  when --lod), -ub 512 cliff (docs say pass -ub 2048; 708 vs 2546 t/s at d64k).

## 2. Performance state (gemma-4-31B q4_0, MI325X)

- pp is depth-flat: +35% at d16k, +119% at d48k vs dense; scales with GPUs.
- tg: 93-100% of dense at practical depths on the hybrid test models; decode
  kernels total ~137us/layer at d16k/top32.
- SEVEN structural kernel alternatives were implemented, measured, and lost to
  the shipped config (SPLITS/NTILES/TILE variants, HPT=4, 2 pipelining variants,
  LDS-staged GQA sharing `GGML_LOD_SPLIT=lds`, persistent mega-kernel
  `GGML_LOD_SPLIT=mega`). Both big variants remain in-tree behind env switches
  for ablation. Conclusion: the plain column loop is latency-bound and
  compiler-optimal; do not attempt further micro-restructuring without new
  evidence (see memory notes; measure with `LOD_BENCH=1 test-lod-attention`,
  `LOD_BENCH_TOP`, `LOD_BENCH_P`).

## 2b. Diagnostics added

- `LLAMA_LOD_CHECK=1` (needs `-v`): at every prefill, recompute all complete page
  sums from the cache cells and compare against the maintained values; prints
  `layer L page P head H dim D: maintained X vs recomputed Y (n_past, sums_pos)`.
  `LLAMA_LOD_CHECK=2` corrupts one page first so you can prove the detector fires.
  Validated both ways (clean run reports "verified", deliberate corruption is
  caught exactly). This exists because two plausible root causes for the server
  multi-turn collapse (cross-stream graph reuse; checkpoint restore dropping the
  sums) were both fixed defensively but NEITHER reproduced the symptom - stop
  guessing mechanisms and let the detector name the culprit.
- `lod-coverage.sh` - the benchmark mk3 could not be: 12 requirements scattered
  through the context, all needed in ONE long generation, scored by how many
  appear. mk3 saturates (every config 3/3 at MK3_N=128) because it asks for one
  fact once; coverage discriminates (dense 12/12, LoD 11/12 at top32).

## 2c. Per-layer budget search (`llama-lod-search`) - measured, and the failures matter

Two generations of this tool exist. Both are in the binary; only the second is trustworthy.

**(a) profile-based allocation (`--target ...`)** - `LLAMA_LOD_PROFILE=1` emits, per layer,
the true dense attention mass per page and the selector's score per page, so ONE dense pass
yields capture(k) for every k. The allocation walks the concave envelope and equalises the
worst layer. Three allocation rules were built and killed by validation before one looked
right - keep these, they are the interesting part:
1. "smallest k with capture >= 0.95" -> every layer takes ~86% of all pages. The missing
   idea is that the baseline is RANDOM selection (k/P), not 1.0.
2. local marginal-lift knee -> stops on a plateau that precedes a cliff (layer 59 goes
   0.155 at k=32 to 0.585 at k=64). The concave envelope fixes it: a hull segment's slope
   is the true value of the pages it spans.
3. profiling the LAST query only -> coverage 3/12, WORSE than uniform with half the budget.
   Prefill shares one page set across 2048 queries; the last query's mass is dominated by
   recent context that the exact tail already reads. Sampling 32 queries and max-pooling
   the score over queries and heads (the selector's own pooling) fixed it.
   And then rule 4 killed the whole approach: repeated draws of lod-coverage put the
   capture-optimal split at 9.33/12 mean against 10.50/12 for plain uniform at the same
   total. The single-draw 12/12-vs-11/12 that made it look like a win was noise. Capture
   is a diagnostic, not an objective.

**(b) measured search (`--probes ... --search sens|greedy`)** - the model is loaded once and
a context is built per candidate budget (`llama_init_from_model` + `llama_free`), so a
candidate costs ~10 s at 24k on gemma4-31B. Objective: retrieval log-probability, i.e.
teacher-force "<key> = " after the document and read the log-prob of the planted value
(`lod-coverage.sh` writes the probe sidecar). Deterministic, monotonic in budget, ~4 nats of
range: top4 -3.94, top16 -2.36, top32 -1.63, top64 -0.99, top128 -0.0002, dense -0.0000.
`--search sens` starves one layer at a time; on gemma4 layer 23 costs +0.81 nats and the
median layer +0.17, so the layers really are unequal - the premise holds.

THE failure to remember: optimising that objective alone drives everything into layer 23
(248 of 320 pages, all others at the 8-page floor), reports -0.0011 - dense-level retrieval
at a uniform-level budget - and produces **2/12 on coverage and 1/3 on needle** against
11/12 and 2/3 for plain uniform. Copying a visible value needs one layer with reach;
generating 400 coherent tokens needs every layer to still read context. Hence `--guard N`:
the last N document tokens are scored as ordinary next-token prediction out of the SAME
prefill the probes run on (holding them out instead changes the probe context and moves the
sensitivity ranking - that was a real bug, briefly shipped), and `--guard-tol` vetoes any
candidate that loses more than that many nats there.

Lesson, now paid for twice: a cheap objective must be validated against the expensive
benchmark before it is trusted, and the validation has to include a task the proxy does not
resemble.


## 2d. BUG - the COMPOSED decode path asserts (LLAMA_LOD_FUSED unset)

Found while building the needle objective into `llama-lod-search`, and initially
misdiagnosed by me as a "2..8 token batch" problem - it is not about the batch size.

A decode graph built with `kP > 0`, `use_fa == 0` and `use_fused == 0` aborts in
`build_attn(llm_graph_input_attn_lod*, ...)` with
`GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3)`. Exact failing geometry, from
`LLAMA_LOD_DBG_SHAPES=1` (added for this, prints one line per layer per graph so the last
line before an abort names the failing case):

    il=5 n_tokens=1 prev_end=26582 P_full=415 P_cap=512 kP=32 n_exact=65 catchup=0 fa=0 fused=0

Nobody sees this in normal use because `--lod` sets `LLAMA_LOD_FUSED=1`, so decode takes the
fused kernel. It bites anything that enables LoD by setting `LLAMA_LOD` directly (as this
tool did) and anyone running the documented `--lod-fused 0` ablation with a real generation
after a long prompt. The composed read is supposed to be the reference path, so this is
worth fixing rather than papering over.

Reproduce: `LLAMA_LOD=1 LLAMA_LOD_TOP_PAGES=32` (deliberately WITHOUT `LLAMA_LOD_FUSED`),
prefill ~26k tokens, then decode one token.

## 2e. `--niah` - the end-to-end needle objective

`--niah file=value,...` prefills each needle document, checks whether greedy decoding emits
the planted value, and returns the pass count (ties inside a tier broken by the log-prob of
the wanted string). The asset now carries ALL TWELVE planted keys - it was six while this
was being built, and every conclusion drawn on the six-document scale had to be withdrawn
(2c). Cost is ~2 min per candidate at twelve documents; do not trim it to save time, the
resolution is the whole point.

Portability: RESOLVED, one code path for both families. On the twelve-document set
gemma4-31B gives 3/12 at top 32 and 6/12 at top 128; qwen3.6-27B gives 8/12, 10/12 and
12/12 at top 32/64/128, reaching dense (12/12) while reading 31% of the context. Three generic fixes were
needed, and it is worth knowing all three because each one silently produced a plausible
wrong number first:
1. `setenv("LLAMA_LOD_FUSED", "1")` - `--lod` does this, setting `LLAMA_LOD` alone does not,
   so the tool was measuring (and crashing on) the composed decode path nobody runs.
2. positions from `llama_memory_seq_pos_max()`, never from the token count - M-RoPE (qwen)
   advances position faster than one per token.
3. no rewind-and-replay - M-RoPE requires each batch to start after the stored max position,
   so the "rewind and teacher-force" pattern cannot work there at all. One forward pass per
   document; read pass/fail and the tie-break out of it.
An exact-greedy-match rule was tried and rejected: fine on gemma4, flat on qwen3.6 because
of answer phrasing. Model-specific rules disguised as metrics are the recurring failure in
this whole line of work.

Result on qwen3.6-27B (16 LoD layers, uniform 32 = 512 pages): the sensitivity signal is
strong - layer 19 costs 2 documents, 31 and 11 cost 1 each, and layers 3/7/23 score NEGATIVE
(starving them GAINS a document). At 512 pages the searched split wins mk3 (2/3 vs 1/3),
loses coverage (11/12 vs 12/12) and ties the needle count - no clear gain. The reason is
budget, not shape: at 512 pages the peaks and the floor compete (floor 8 -> coverage 11/12
needle 4/6; floor 24 -> coverage 12/12 needle 2/6). Give it both - peaks 128 on 11/19/31
with a floor of 24, 696 pages - and it reaches needle 5/6, coverage 12/12, mk3 2/3, against
3/6 for the equal-cost uniform 44 (704 pages) and matching uniform 64 (1024 pages) on every
metric: 32% cheaper at equal quality. Lesson: a search run below the budget where peaks and
floor stop competing measures the competition, not the allocation.

HARNESS BUG found here and fixed: lod-needle-mk3.sh generated 8 tokens by default, and
qwen3.6 opens a thinking block on this prompt (`--reasoning-budget 0` is inert because
-no-cnv means no chat template). Every configuration including DENSE scored 0/3, which I
first read as "mk3 does not work on qwen" and used to justify a "keep uniform" verdict. At
MK3_N=256 the rows separate: dense 3/3, top32 1/3, searched 2/3, top128 3/3 - i.e. qwen
really does lose retrieval at top32 and recover at top128, which was invisible before. The
default is now 256 (and the output window 4000 bytes). A benchmark that returns the same
number for every configuration is a broken instrument, not a result - check dense first.

Two operational notes that cost hours here, both mine rather than the code's:
- Always `stdbuf -oL` or redirect to a file. A pipe into `sed`/`grep` block-buffers, and a
  perfectly healthy run then shows no output for minutes and looks hung.
- Do not pre-create cancel markers in `~/.cluster-run/*/cancel/` for job ids you guessed:
  the runner reuses ids, and a stale marker kills the next job instantly. Several "crashes"
  in this work were that, plus my own `kill`s (the job logs show `exit_code=143 signal=15`,
  i.e. TERM from outside, not an abort).

What it found on gemma4-31B (baseline: all ten LoD layers at 32 pages = 2/6 answered;
each layer starved to 8 in turn), with the teacher-forced probe's ranking beside it:

    layer 11  -2.02   needle rank 1   probe rank 7
    layer 29  -2.01                2               6
    layer 23  -2.00                3               1
    layer 41  -1.01                4               3
    layer 35  -1.00                5               5
    layer  5  -0.02                6               4
    layer 17  -0.02                7              10
    layer 59  -0.01                8               2
    layer 47   0.00                9               8
    layer 53  +0.00               10               9

Five layers carry the retrieval, five are free - the premise holds. Running the search on
this objective (15 evaluations) puts 104/104/56 pages on layers 11/29/23 and 8 on everything
else - the same 320 pages as uniform 32 - and that config validates: needle 2/6 -> 4/6,
coverage 11/12 -> 12/12 (dense, at a quarter of uniform-128's pages), mk3 3/3 unchanged
(mk3 at 32k is saturated on gemma4 once the generation length is fixed - see 2f).
That is the one allocation in this whole line of work that beat uniform end to end, and it
only appeared when the objective WAS the benchmark. But the two rankings
disagree badly (11: first vs seventh; 59: eighth vs second), so the probe objective must not
be used to allocate budget. It is a smoke test, nothing more. Note also that uniform 32 is a
cliff for this model: 2/6 at baseline, 0/6 if almost any middle layer is starved, so a
search wants a baseline with headroom (top 64-96).
`llama-lod-search` now frees the discovery context (`common_init_result::free_context()`,
added for this) before the search loop, so only one context is live at a time.

## 2f. gather quality is not monotone in the budget - mask is

Measured, reproducible (temp 0, identical on repeats), gemma4-31B coverage at 24k:

    budget   pages   gather   mask
    top 40    400    12/12    12/12
    top 48    480     7/12    12/12
    top 56    560     7/12    12/12
    top 64    640    12/12      -

At top 48/56 the model writes seven settings and skips requirements 6-10 - a contiguous
region of the document - while still answering the ones after it, so it is not truncation.
One page set serves the whole ubatch in gather mode, and at those budgets the set covers no
page in that region. mask (per 32-query block) does not have the failure.

This explains a large part of the measurement noise in sections 2c-2e: a coverage difference
of 1 in 12 between two configurations is far inside the swing caused by moving the budget by
8 pages per layer. Any tuning done on gather needs a sweep, not a point measurement.

## 2g. stratified page selection - implemented, measured, REVERTED

Response to 2f: replace the global top-k with "best kP/r pages inside each of r slices of the
page axis". Same pages read, no shape change (static graph and its reuse unaffected), no
measurable throughput cost (2605/2495 t/s at r=1 against 2538/2596 at r=2).

    gemma4-31B 24k        r=1        r=2        r=4
    coverage top 48        7/12      12/12       6/12
    coverage top 56        7/12      12/12      12/12
    needle/12 top 32         3/12       3/12       1/12
    needle/12 top 64         4/12       2/12       1/12

Fixes the coverage hole, costs needle monotonically in r. Reverted: the point of touching
gather was to move it TOWARDS mask's needle behaviour, and this moves the other way. The code
is gone; keep the numbers so it is not rebuilt.

Implementation note if something similar is ever attempted: `ggml_add`/`ggml_cast` on I32
index tensors silently produce garbage (first attempt scored 0/12 everywhere). The working
form keeps the top-k indices slice-local and resolves them with a batched `ggml_get_rows`
over a slice-shaped view of the page rows - no index arithmetic at all.

## 2h. why gather cannot cheaply reach mask's needle quality (MEASURED)

The gap: gemma4-31B needle on 12 documents, gather 3/12, 4/12, 6/12 at top 32/64/128 against
mask 3/12, 10/12, 12/12. mask matches dense at top 128 and reaches 10/12 at top 64 where
gather needs top 256.

The cause is entirely query granularity, established by widening mask's block:

    mask top 64, block  8  16  32(default) 128  512  2048
    needle/12           8   9      10        5    4     2

At block 2048 - one page set for the whole ubatch, i.e. gather's granularity - mask lands on
gather's score. Note 32 is an optimum, not a floor: at 8 queries the page scores come from
too few queries and the selection turns noisy. Block size is free in mask mode (1990/2066 t/s
at QB 8 against 2063/1971 at QB 32), the full-span read dominates.

Why gather cannot buy it:
- one gather per 32-query block = 64 blocks at ub 2048, each needing its own copy of the
  shared exact + summary tiers ~ 2.4 GB per layer. Not viable.
- one gather of the union of all blocks' selections: the union was measured at 47-82% of all
  pages, i.e. mask's cost without mask's simplicity.

So within the current tier structure there is no cheap path. The lever that has not been
tried is the tier structure itself - richer page summaries (more than one mean per page, or a
second-level summary) so that a coarse selection loses less information. That, not selection
granularity, is where a further attempt should go.

## 2i. per-block gather - the recipe, and why it was not built

Kept because the tensor plumbing was worked out and is not obvious. `ggml_get_rows`
is batched on the index tensor, so `get_rows(k_rows[D*ps, P_full], sel[kP, n_blk])`
returns `[D*ps, kP, n_blk]` directly - no index arithmetic. `ggml_flash_attn_ext`
requires `q->ne[3] == k->ne[3]` and broadcasts the mask over ne[3], so block-diagonal
attention IS expressible: `q4 = [D, QB, Hq, n_blk]`, `kk4 = [D, N_set, Hkv, n_blk]`,
`mask4 = [N_set, QB, 1, n_blk]`, and the attention FLOPs are unchanged.

What kills it is 2h: the granularity has to be 32 queries, so n_blk = 64 at ub 2048,
and the shared exact + summary tiers have to be replicated per block (~2.4 GB/layer).
At QB 256 the replication is affordable (~300 MB/layer) but the quality is not there
(mask itself only scores 5/12 at block 128).

## 3. OPEN INVESTIGATION - the multikey3 retrieval failure (IMPORTANT)

Benchmark: `lod-needle-mk3.sh <model> <6k|32k|100k> <ctx>` - 3 keys at 10/50/90%
depth, one question each (assets /home/mose/Projects/llm/lod-mk3-*).

Established facts (all measured, in order):
1. 6k/top16 = 3/3 (STANDARD REGRESSION - keep it green). top8 = 1/3.
2. 32k/100k: far keys fail at ANY practical budget (100k needs top1024 of 1560
   pages!); WHICH key passes varies with haystack seed (lottery).
3. Distance hypothesis REJECTED (32k: nearest key fails, mid key passes).
4. Rank instrumentation (env `LLAMA_LOD_DBG_SCORES="p1,p2,.."`, temp code in
   lod-attn.cu, decode only, needs -n >= 2): needle pages rank TOP-10 in nearly
   every layer at decode. SELECTION IS NOT THE PROBLEM.
5. Full expansion at 32k == dense == correct answer. MACHINERY IS NOT THE
   PROBLEM.
6. Tail-pooled prefill selection (last-512 queries) did NOT fix it and
   regressed 6k (default reverted; opt-in `LLAMA_LOD_SELPOOL=tail`).
7. Failure mode: the model CONTINUES THE NARRATIVE instead of answering -
   the first generated token comes from PREFILL logits, so the derailment
   happens during prompt processing, not decode.

RESOLUTION for 32k (2026-08-10/11): the counterpart LoD1 measurements showed the
query axis is everything and the head axis is nearly free - per-32-query-block
selection captures ~0.72 of the attention mass vs ~0.31 for one set shared across
the whole ubatch. Transplanting that granularity (mask-direct prefill) makes 32k
retrieval MONOTONIC in the budget: 0/3 -> 2/3 -> 3/3 at top 16/64/128, where
gather is a lottery (2/3, 1/3, 3/3 at 32/64/128 - verified on a clean run).

STILL OPEN at 100k: per-block selection did not rescue it (1/3 at top 64/128/256)
and neither did the selection-only region tier. Established: selection ranks are
excellent at decode, full expansion == dense, so this is score SNR at ~2000
candidates - the page sums of far pages stop carrying a usable signal between
roughly 30k and 90k of query-key distance. The remaining untested hypothesis is
that the score should be computed on PRE-RoPE sums (rotated keys cancel across 64
positions, and the surviving low-frequency dims get pseudo-random phase at large
distance). That needs a second set of sums maintained from pre-RoPE K plus
model-side wiring of pre-RoPE Q; the read math stays untouched.

Calibration trap to remember: pages = tokens/64 and tokens ~ bytes/4.8 for these
haystacks (NOT 4.2) - always sanity-check against the n_past the instrumentation
prints.

## 4. Mask-direct prefill - DONE

`--lod-prefill mask` (env `LLAMA_LOD_PREFILL=mask`). Flash-attention over
`[page means | raw cache span]`, the entire selection in the additive mask, so
each 32-query block picks its own pages at no extra read cost. Mean tensors live
in the KV cache in cache precision (F16 or q8_0) and are refreshed from the sums
every prefill ubatch. Validated: 6k top16 3/3, 32k top128 3/3 (F16 and q8_0),
budget-monotonic, and the span is capacity-sized so the graph shape never changes
(the causal formula does not contain the span, masked columns contribute exactly
0.0, and the cache is zero-initialised, so padding cannot perturb the result -
confirmed 3/3 on all three quality points).

The honest cost, measured across depth: mask reads the token tier in full, so its
prefill is dense-speed and decays like dense (2461 -> 1239 t/s over 16k -> 82k;
dense 2519 -> 1308; gather 2816 -> 2615). It is a quality mode, not a speed mode,
and it is NOT the right default - gather is the only mode that delivers the
depth-flat prefill LoD exists for.

Known structural limit: flash-attention's mask scan can only truncate a fully
masked SUFFIX (it never skips interior holes), which is why the mean tier sits at
the FRONT of the KV axis. Interior sparsity would need a fattn kernel with a
per-tile KV block list - upstream-scale work.

## 5. Ablation and debug switch reference

- `LLAMA_LOD`, `--lod`, `--lod-top-pages`, `--lod-page-size`, `--lod-sel`
- `--lod-top-pages-decode` / `LLAMA_LOD_TOP_PAGES_DECODE` per-phase budgets
- `--lod-prefill gather|mask` / `LLAMA_LOD_PREFILL` prefill mode (default gather)
- `LLAMA_LOD_FUSED=0` composed decode; `LLAMA_LOD_SEL=head` per-KV-head sets
- `LLAMA_LOD_QB=N` mask-mode query-block size (default 32; quality holds to 64,
  drops at 128; speed is invariant, so there is no reason to raise it)
- `LLAMA_LOD_SPAN=N` mask-mode growing span in N-token steps (default: capacity,
  which is what keeps the graph shape static - only for ablation)
- `LLAMA_LOD_CHECK=1|2` page-sum integrity check (2 = self-test); needs `-v`
- `GGML_LOD_SPLITS=16|32|64|128` split-K width (default: device-adaptive, targets
  4 split blocks per CU); `GGML_LOD_SPLIT=lds|mega` alternative decode kernels
- `LLAMA_LOD_SELPOOL=tail` tail-pooled prefill selection (regressed 6k; off)
- `LLAMA_LOD_DBG_SCORES="p,..."` decode selection-rank dump (temp code)
- `LOD_BENCH=1 [LOD_BENCH_TOP=k] [LOD_BENCH_P=p] [LOD_BENCH_D=d] test-lod-attention`
- `GGML_CUDA_DISABLE_GRAPHS=1`, `LLAMA_GRAPH_REUSE_DISABLE=1` infra ablations

## 6. Methodology lessons (paid for in wrong turns)

- Node performance swings +-10% run-to-run: only alternating same-condition
  pairs are trustworthy; single-shot A/B lied to us repeatedly.
- Rebuild-vs-reuse and gallocr shape stability interact with pipeline
  parallelism; "conditionally created inputs must be conditionally consumed"
  (6 crashes from this rule).
- Cross-kernel outputs differ at float ULP; test tolerance 1e-6, same-kernel
  paths stay bitwise.
- Needle pass/fail through a 31B model is a noisy 1-bit oracle with huge seed
  variance; instrument continuous quantities (ranks, scores) before theorizing.
  A benchmark that saturates (every config 3/3) is measuring nothing - mk3 at
  MK3_N=128 does exactly that, which is why `lod-coverage.sh` exists.
- ALWAYS run the control. A 25-turn conversation showed token stutter under
  `--lod` that looked exactly like corruption; dense produced the same stutter.
  Twice in one session a control killed a wrong conclusion.
- Calibrate a new benchmark against dense BEFORE reading any LoD number from it.
  The first coverage run scored 0/12 for dense AND LoD - the generator's
  tokens-per-line estimate was 3x off and the prompt exceeded the context.
- GPU jobs go through `CLUSTER_GPUS=<ids> bash simple-run.sh ...`. Scripts that
  invoke `llama-cli` themselves (`lod-needle-mk3.sh`, `lod-coverage.sh`) must be
  wrapped WHOLE - `CLUSTER_GPUS=0 bash lod-needle-mk3.sh ...` silently runs
  locally with CLUSTER_GPUS ignored. Also: `simple-run.sh` re-quotes with
  `printf '%q'`, so inline multi-line shell breaks; put the sweep in a file.
- Never put a `pgrep -f`/`pkill -f` pattern in a command line that contains the
  same string: it matches the shell running it and kills the session.
- Library log output (`LLAMA_LOG_*`) is suppressed by llama-cli and llama-server
  unless `-v`; a diagnostic that "does not fire" may just be invisible.
