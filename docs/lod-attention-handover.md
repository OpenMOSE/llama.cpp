# LoD attention - handover notes (as of 2026-08-10)

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
  -> 4246 (2) -> 6586 (7). Key coupling: under pipeline parallelism the graph
  must take the REBUILD path (llama.cpp synchronizes reused graphs per ubatch);
  single GPU keeps reuse. See `can_reuse` in llama-graph.cpp.
- State machine under mutation: suffix truncate = page-floor rewind + physical
  row zeroing + lazy catch-up refold. This fixed real corruption after server
  prompt-cache reuse AND after MTP rollbacks (folds are read-modify-write; a
  bare watermark rewind double-counts).
- Multi-slot server (-np N, per-stream sums, dense fallback for mixed batches),
  MTP (gemma assistant gguf + qwen3.6 built-in), quantized KV via composed path.
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

Next steps (in order):
- Dump the PREFILL selection contents: prefill uses graph-side top_k (named
  `lod_sel-<il>` via cb), invisible to the decode instrumentation. Use the
  eval-callback tool (tools/, llama-eval-callback) filtered to `lod_sel` on a
  32k q1 run: are needle pages selected in the FINAL segment? In MIDDLE
  segments?
- Sweep top_pages 128/192/256 at 32k for the flip point.
- Hypothesis space still open: (a) middle-segment read quality corrupts the
  running representation the question builds on; (b) something specific to the
  final small segment (344 tokens at 32k: 26968 = 13*2048 + 344); (c) summary
  -mass side effects on instruction-following.
- Note the token/byte calibration trap: pages = tokens/64, tokens ~ bytes/4.8
  for these haystacks (NOT 4.2) - always sanity-check against n_past printed by
  the instrumentation.

## 4. In-flight work: mask-direct prefill (`LLAMA_LOD_PREFILL=mask`)

Goal: replace the per-layer leaf-gather chain (~15 nodes, 50-100MB/layer/ubatch)
with fattn over [cache span | page-mean columns] + pure-mask selection. Design
v3 is final (see memory notes + spec): causal base mask is ONE meta-driven
formula `log(step(P_rt*ps + c0 + row - col))`, per-layer additions are a
page-kill additive term + the existing mean-column silencing machinery; K/V =
concat(cache_span, k_mean view); span rounded to 16k steps; F16 caches only.

Status: Step A (k_mean/v_mean tensors + getters) and Step B (mean refresh from
sums each prefill ubatch, input `lod_meanidx`) are DONE and validated no-op.
Step C (the read branch) and Step D (A/B: PPL parity, mk3, pp 1/2/7 GPU,
CPU graph-build time) remain.

## 5. Ablation and debug switch reference

- `LLAMA_LOD`, `--lod`, `--lod-top-pages`, `--lod-page-size`, `--lod-sel`
- `LLAMA_LOD_FUSED=0` composed decode; `LLAMA_LOD_SEL=head` per-KV-head sets
- `GGML_LOD_SPLIT=lds|mega` alternative decode kernels (slower; ablation)
- `LLAMA_LOD_SELPOOL=tail` tail-pooled prefill selection (regressed 6k; off)
- `LLAMA_LOD_PREFILL=mask` mask-direct prefill (incomplete: Step C pending)
- `LLAMA_LOD_DBG_SCORES="p,..."` decode selection-rank dump (temp code)
- `LOD_BENCH=1 [LOD_BENCH_TOP=k] [LOD_BENCH_P=p] test-lod-attention` op harness
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
- GPU jobs go through `CLUSTER_GPUS=<ids> bash simple-run.sh ...`; a direct run
  lands on CPU and looks like a hang.
