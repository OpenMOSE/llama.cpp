# LoD2: Key-Value Means sparse attention (experimental)

LoD2 is a training-free sparse read for the full-attention layers of a model.
It is **not** LoD1 with different budgets: LoD1's coarse tier is a fixed
positional page hierarchy, LoD2's is a set of **content-addressed clusters**
("KV means"), and the exact tokens are reached *through* those clusters rather
than through positions.

Reference: *Key-Value Means*, arXiv:2605.09877.  The algorithm-level contract
this port satisfies is `lod2-port-spec.md`; the executable restatement is
`lod2-pytorch/lod2_ref.py`; the engineering log, including every measurement and
every rejected hypothesis, is `lod2-port-handover.md`.

## What it computes

Three disjoint branches over the history, merged exactly by logsumexp.

1. **coarse** - one entry per state slot: key = the slot's mean key, value = its
   mean value, logit bias = `log(count)`.  A slot the query has *opened* is
   removed from this branch.
2. **local** - the most recent `--lod2-local` tokens, exact and causal.
3. **routed** - the top `--lod2-routes` slots by coarse logit.  Each opened slot
   contributes one exact page of `--lod2-page` tokens plus one count-corrected
   residual standing for the rest of that slot.

Because an opened slot is silenced in branch 1 and reintroduced in branch 3 with
the same total mass, **opening every slot reproduces dense attention exactly**.
That is refinement, not approximation, and it is the property the whole
verification chain rests on.

The state holds `--lod2-growth * sqrt(N)` slots, so the read grows as the square
root of the context.  At 32k with the defaults: 2896 slots + 512 local + 8*16
routed leaves, about **10.8% of the context**.

## Usage

```sh
llama-cli    -m model.gguf --lod2 [--lod2-routes 8] [--lod2-growth 16] ...
llama-server -m model.gguf --lod2 --port 8080 -np 4
```

The flags work in every tool that uses common args (cli, server, bench,
perplexity).  Each one sets the environment variable the engine reads, so the
two are interchangeable.

| flag | env | default | meaning |
|---|---|---|---|
| `--lod2` | `LLAMA_LOD2=1` | off | enable for full-attention layers |
| `--lod2-growth F` | `LLAMA_LOD2_GROWTH` | 16 | slots = `F*sqrt(N)` |
| `--lod2-state-min N` | `LLAMA_LOD2_STATE_MIN` | 256 | floor on slots |
| `--lod2-local N` | `LLAMA_LOD2_LOCAL` | 512 | exact window at generation; multiple of chunk |
| `--lod2-chunk N` | `LLAMA_LOD2_CHUNK` | 256 | absorb step, and the grain of the window boundary |
| `--lod2-routes N` | `LLAMA_LOD2_ROUTES` | 8 | slots opened per query at generation |
| `--lod2-routes-prefill N` | `LLAMA_LOD2_ROUTES_PREFILL` | 3 | same, prompt processing |
| `--lod2-page N` | `LLAMA_LOD2_PAGE` | 16 | tokens per page inside a slot |
| `--lod2-pages-per-slot N` | `LLAMA_LOD2_PAGES_PER_SLOT` | 128 | page-table capacity per slot |
| `--lod2-sink N` | `LLAMA_LOD2_SINK` | 1 | leading slots never routed |
| `--lod2-update N` | `LLAMA_LOD2_UPDATE` | `5*chunk` | columns absorbed per prefill state update |
| `--lod2-layers SPEC` | `LLAMA_LOD2_LAYERS` | all | `"il,il,..."` selects exactly those layers |
| `--lod2-tiled MASK` | `LLAMA_LOD2_TILED` | 3 | tiled prompt kernels: 1 local, 2 coarse, 0 the per-query reference |

The paper's operating point is `-ub 4096`, which makes the prefill window
exactly its 4864 tokens; the prefill chunk *is* the ubatch size.

To confirm your settings took effect, run with `-lv 4`:

```
llama_context: LoD2 attention enabled: state = 16*sqrt(N) (min 256), local = 512,
page = 16, routes = 8 (prefill 3), prefill chunk = 4096, state update = 1280
```

At the default verbosity of 3 no `llama_context` INFO line is printed at all, so
its absence tells you nothing.  **Do not try to confirm the flag by diffing
against dense** - at ordinary sizes LoD2's output *is* dense's output, because
that is what refinement means, so "identical" proves nothing about whether the
flag took.

## Requirements and limitations

* **One KV cache stream per sequence.**  LoD2 indexes the archive by position
  and the cache indexes it by cell, and the two must coincide.  That holds when
  each sequence owns its own stream, which is the default; `--kv-unified` is
  refused with a clear error.  `-np N` works.
* **Quantized KV cache** works (`-ctk q8_0 -ctv q8_0`, and f16/f32).
* **No context shift, no prefix reuse, no speculative rollback.**  Slots hold
  sums, so removing a suffix would need the additions undone.  `seq_rm` drops
  the whole state for that stream and the next ubatch's schedule rebuilds it; a
  hole in the middle is refused.
* Wired into the hybrid Qwen3.5/3.6 graphs (`src/models/qwen35.cpp`,
  `qwen35moe.cpp`).  Other architectures need the same three-line hook.
* Graph reuse is off (`LLAMA_LOD2_REUSE`); it produced wrong output and has not
  been fixed.

## Measured results

Qwen3.6-27B-Q4_K_M, 32k prompt, MI325X, `-ub 4096`.  LoD2 at its defaults.

| | dense | LoD2 |
|---|---|---|
| prompt, 1 GPU | 2705 t/s | 1985 t/s |
| prompt, 8 GPU (pipeline) | 5530 t/s | 4131 t/s |
| generation, 8 GPU | 75.7 t/s | 63.5 t/s |

What `--lod2-tiled` is worth, same run, prompt on 8 GPUs: `0` (per-query
reference kernel) 1998 t/s, `1` (local branch tiled) 3125, `2` (coarse branch
tiled) 2383, `3` (both, the default) 4166 t/s.  All four produce identical
output.

LoD2 is still **behind** dense on this hardware at 32k, and honestly so: it
reads about 2x fewer columns but its kernels are several times less efficient
per column than a tuned flash-attention kernel.  Per generated token the LoD2
attention kernels cost 2.50 ms against dense attention's 1.12 ms.  The gap
closes as the context grows, because LoD2's read is `O(sqrt(N))` and dense's is
`O(N)`; at 32k that has not paid for the constant factor yet.

Quality, Qwen3.5-2B at 8k, needle-in-a-haystack (4 documents per task):

| | dense | LoD2 |
|---|---|---|
| `niah_single_3` | 4/4 | 4/4 |
| `niah_multikey_2` | 4/4 | 4/4 |

## How it is verified

The chain, all reproducible:

```
author's implementation == lod2_ref.py == ggml_lod2_* (CPU) == ggml_lod2_* (HIP) == llama.cpp end to end
```

* restatement vs the author: state bit-identical, output 1.3e-6
* **refinement**: open every slot and LoD2 must equal dense.  5.7e-7 in PyTorch,
  and *identical greedy tokens* end to end
* the op oracle (`tests/test-lod2.cpp`) replays dumped cases through the CPU and
  HIP kernels: counts exact, outputs ~1.3e-6
* the tiled prompt kernels (`--lod2-tiled`) are held to the same bar - identical
  output to the per-query reference kernel, not merely close

```sh
# reference vs author, and the refinement property
cd docs/lod2-pytorch
python check_lod2_ref.py --len 1024 --decode 2 --device cpu --head-dim 64 --query-heads 4 --kv-heads 2
python check_lod2_ref.py --dense-check --len 1536 --device cpu --head-dim 64 --query-heads 4 --kv-heads 2

# op oracle
python docs/lod2-pytorch/dump_lod2_case.py --out /tmp/c.bin --len 1024 --decode 4 \
    --query-heads 4 --kv-heads 2 --head-dim 64 --device cpu
./build/bin/test-lod2 /tmp/c.bin

# end to end refinement: these two must produce identical text
M=/home/mose/Projects/llm/Qwen3.5-2B-BF16.gguf
C="-m $M -f /tmp/p.txt -n 48 --temp 0 -ngl 99 -c 4096 -ub 512 -st -no-cnv --simple-io"
./build/bin/llama-cli $C
./build/bin/llama-cli $C --lod2 --lod2-growth 1000000 --lod2-routes 100000 --lod2-routes-prefill 100000
```

Two traps this port has already fallen into, worth repeating:

* **an "identical" between two empty files is not a pass.**  Always print the
  number of lines compared.  `grep "^Client"` matches nothing in
  llama-parallel's output, because every line carries a timestamp and colour
  codes - strip them and compare the `Response:` lines.
* **at three or more concurrent sequences there is no text-level verdict.**  The
  workload is token-unstable under any FP reordering there: dense against dense
  with only `-ub` changed (512 vs 384) already differs.  LoD2 is verified exact
  against dense at one and two sequences, and structurally asserted beyond that.

## Code map

| path | what |
|---|---|
| `src/llama-lod2.h` | the schedule - one definition, shared by the graph builder and the test |
| `src/llama-graph.cpp` | graph construction: `build_attn_inp_lod2`, `build_attn(..., inp_lod2, ...)` |
| `src/llama-kv-cache.{h,cpp}` | state storage (per stream), and the `seq_rm` rollback rule |
| `src/llama-context.cpp` | parameter parsing and the one-stream-per-sequence guard |
| `common/arg.cpp` | the `--lod2*` flags |
| `ggml/src/ggml-cpu/lod2.cpp` | CPU implementation of both ops - the contract the GPU must match |
| `ggml/src/ggml-cuda/lod2.cu` | HIP/CUDA kernels for both ops |
| `tests/test-lod2.cpp` | replays an oracle case through whichever backend is built |
| `docs/lod2-pytorch/` | the reference implementation, the case dumper, the NIAH driver |
| `docs/lod2-port-spec.md` | the contract |
| `docs/lod2-port-handover.md` | engineering log: measurements, dead ends, next steps |

Two ggml ops carry it: `GGML_OP_LOD2_UPDATE` folds a block of archive columns
into the state (similarity, split, merge, page bookkeeping), and
`GGML_OP_LOD2_ATTN` computes the three branches and their logsumexp merge.  The
graph emits one update chain and one attention op **per sequence group** in the
ubatch, which is what makes `-np` work without any kernel knowing about streams.

## Choosing parameters

* `--lod2-routes` is the accuracy knob.  It is what refinement is parameterised
  by: raise it and the output converges on dense, monotonically.  8 at
  generation is the paper's setting.
* `--lod2-growth` trades state size against coarse-branch resolution.  It also
  sets the decode cost, since the coarse branch and the routing logits both walk
  every slot.
* `--lod2-local` is the exact recent window.  It is cheap and it is what protects
  local coherence; leave it at 512 unless you have a reason.
* `--lod2-tiled 0` selects the per-query reference kernel.  Use it if you suspect
  the tiled path; it is slower by about 2x on prompt processing and is expected
  to produce identical output.

## Tuning and debug knobs (environment only)

These are for bisecting the port, not for running it, and are deliberately not
exposed as flags: `LLAMA_LOD2_NS`, `NSMUL`, `PWPB`, `SIMMT`, `SIMB`, `RM`, `CT`,
`CM`, `LM`, `KVH`, `HPW`, `LOCALDS`, `REUSE`, and `LLAMA_LOD2_DBR` - a branch
mask on the decode kernel (1 coarse, 2 local, 4 routed) that produces wrong
output on purpose.  `DBR` is also the way to prove the decode path is actually
running: masking it must change the output.

## Known gaps

* Generation is 63.5 t/s against dense's 75.7 at 32k on the 27B.  The remaining
  cost is dominated by wave-wide reductions: `__shfl_xor_sync` at width 64
  lowers to `ds_bpermute_b32`, and `lod2_k_sim`'s inner loop is 24 of them
  against 16 FMAs.  Removing them means giving one lane a whole dot product,
  which changes the summation order and would cost the bit-stable refinement
  signal - that trade has not been taken.
* Prompt processing is 1985 t/s against dense's 2705 at 32k on one GPU.
* Multi-GPU prefill scales like dense (1.00 / 1.48 / 2.10 for 1 / 2 / 8 GPUs)
  after the graph-stability fix, so the remaining gap is per-device efficiency,
  not scaling.
* INT4 leaf storage (`kv_bits=4` in the paper) is not implemented.  It is a
  storage format and the author measured it as lossless on 32k NIAH.
* NIAH has only been run at 8k with four documents per task.  It is a sanity
  check, not a quality evaluation.
