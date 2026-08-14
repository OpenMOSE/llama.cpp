# LoD2 port spec (llama.cpp)

This is the contract the llama.cpp port must satisfy.  It is written from the
author's final implementation, not from memory of LoD1:

* algorithm of record: `/home/mose/Projects/KVM-paper/model/`
  (`pytorch_lod_attention.py`, `pytorch_lod_attention_fast.py`,
  `pytorch_lod_attention_paged.py`, `triton_lod_attention.py`,
  `triton_lod_engines.py`), upstream commit `430c89c`
* wrapper actually evaluated: `/home/mose/Projects/llm/LoD2-Qwen3.5-2B`
  (`lod2.py`, preset `recursive_int4`)
* paper: *Key-Value Means*, arXiv:2605.09877

`docs/lod2.md` is the user-facing document (flags, requirements, measured
results, code map); `docs/lod2-port-handover.md` is the engineering log.

LoD2 is **not** LoD1 with a different budget.  LoD1's coarse layer is a fixed
positional page hierarchy.  LoD2's coarse layer is a set of **content-addressed
clusters** ("KV means"), and the exact leaves are reached through those
clusters, not through positions.  None of the LoD1 machinery in
`src/llama-graph.cpp` is reusable except the general lesson about static graph
shapes and capacity padding.

---

## 1. What LoD2 computes

Three disjoint branches over the history, merged exactly by logsumexp.

1. **coarse** - one entry per state slot.  Key = slot mean key, value = slot
   mean value, logit bias = `log(count)`.  Slots that a query has *opened* are
   removed from this branch (`-inf`).
2. **local** - the most recent `local_len` tokens, exact, causal.
3. **routed/exact** - for each query, the top `R` slots by coarse logit.  For
   each opened slot, read **one exact page of 16 tokens** plus **one
   count-corrected residual summary** standing for the rest of that slot.

Because an opened slot is silenced in branch 1 and reintroduced in branch 3
with the same total mass, opening every slot reproduces dense attention
exactly.  This is refinement, not approximation.

Read volume at 32k (Qwen3.5-2B, `state_growth_factor=16`): state 2896 +
local 512 + leaves 8*16=128, about 10.8% of the context.

---

## 2. Configuration of record (preset `recursive_int4`)

`PagedLODConfig` + `KernelRecursivePagedLODAttention`:

| name | value | meaning |
|---|---|---|
| `chunk_size` / `chunk_len` | 256 | state-update block, local-window rounding |
| `local_window` / `local_len` | 512 | decode exact window |
| `state_growth_factor` | 16.0 | state budget `16*sqrt(N)` |
| `state_min_size` | 256 | floor on the state budget |
| `protected_prefix` / `sink_len` | 1 | slot 0 is an attention sink |
| `exclude_sink_from_routes` | true | slot 0 is never opened |
| `separate_sink_cache` | false | the sink lives inside the state |
| `max_routes` | 8 | hard cap |
| `two_level_topk` | 8 | routes at decode |
| `prefill_two_level_topk` | 3 | routes at prefill |
| `page_size` | 16 | tokens per logical page |
| `kv_bits` | 4 (or 0) | INT4 leaf archive |
| `quant_group_size` | 32 | INT4 group scale |
| `page_summary_quant_bits` | 8 | page summaries stored INT8 |
| `prefill_chunk_len` | 4096 | `16 * chunk_size` |
| `prefill_local_len` | 4864 | `prefill_chunk_len + local_window + chunk_size` |
| `prefill_state_update_len` | 1280 | `5 * chunk_size` |
| `decode_state_update_len` | 256 | |
| `leaf_inline_pages_per_slot` | 128 | posting-list capacity before hashing |

Derived: `exact_lookback = prefill_local_len - prefill_chunk_len = 768`.

---

## 3. State

Per (sequence, layer, kv head).  Batch is 1 in the reference and in phase 1
of this port.

```
state_k   [H_kv][S_cap][D]    sum of member keys      (NOT the mean)
state_v   [H_kv][S_cap][Dv]   sum of member values
counts    [H_kv][S_cap]       f32 member count; <= 0.5 means "inactive"
state_len                     scalar, number of live slots (host-side)
coverage                      scalar, tokens absorbed into the state
```

Means are recovered by division: `mean_k = state_k / max(count,1)`.  Sums are
used because merging is then plain addition, which makes the recurrence
append-only.  The price is that removing a suffix requires physically undoing
the additions - relevant for prefix reuse and speculative rollback.

`S_cap = round_up(desired_state_len(N + chunk_len), chunk_len)` where

```
desired_state_len(ctx, avail, cur) =
    max(cur, min(max(floor(16*sqrt(ctx)), 256), avail))
```

### 3.1 Page tables (the leaf index)

```
slot_pages   [H_kv][S_cap][128]  int32  page ids owned by each slot, -1 empty
slot_lengths [H_kv][S_cap]       int32  tokens in each slot
next_page    [H_kv]              int32  page-pool bump allocator
page_indices [H_kv][P_cap][16]   int32  token position of each page lane, -1 empty
page_sum_k   [H_kv][P_cap][D]           sum of the page's keys
page_sum_v   [H_kv][P_cap][Dv]          sum of the page's values
page_counts  [H_kv][P_cap]       int32  tokens in the page
```

`P_cap = ceil(seq_capacity/16) + S_cap`.  Slots with more than
`128*16 = 2048` tokens spill into a hash table in the reference; the port may
instead grow the inline list, but must not silently drop pages.

`page_indices` holds **positions into the leaf archive**.  In llama.cpp the
leaf archive *is* the KV cache, so these are KV cache cell indices.  No second
copy of K/V is needed for `kv_bits=0`.

---

## 4. State update (the KV-means recurrence)

Called with an overflow block of `M` tokens (`M = prefill_state_update_len`
during prefill, `decode_state_update_len` during decode; the first block is
`chunk_len`).

**Initialization** (`state_len == 0`): the first `chunk_len` tokens each become
their own slot; `count = 1`, `owner[i] = i`.

**Update** (`state_len > 0`):

```
desired   = desired_state_len(ctx_len, available_context, state_len)
n_append  = min(max(desired - state_len, 0), M)

sim[m][s] = dot(k_m, mean_k_s)                      # NO scale, NO log(count)
sim[m][s] = -inf where count_s <= 0.5
protected_score[m] = max over s < sink_len of sim[m][s]
sim[m][s] = -inf for s < sink_len                   # sink is never a merge target
best_score[m], best_slot[m] = max over s of sim[m][s]
select[m] = max(best_score[m], protected_score[m])

order      = argsort(select, ascending)             # most novel first
append_idx = sort(order[:n_append])                 # ascending token order
merge_idx  = sort(order[n_append:])
```

* appended tokens occupy slots `state_len .. state_len+n_append-1`
  with `key_sum = k`, `value_sum = v`, `count = 1`, `owner = that slot`
* merged tokens go to `dest[m] = argmax over the *union* of the old
  non-protected slots and the newly appended slots`, evaluated against the
  **pre-merge** means.  Concretely: `dest = best_slot[m]` unless
  `max_j dot(k_m, k_appended_j) > best_score[m]`, in which case it is that
  appended slot.
* all merges are applied simultaneously:
  `key_sum[s] += sum of k_m with dest=s`, likewise `value_sum`, `count += 1`
* `owner[m] = dest[m]`

`ctx_len` differs between the two prefill loops:

* main loop: `ctx_len = query_begin + update_end - bswa_begin
             = update_end + exact_lookback`
* decode-boundary loop: `ctx_len = min(prefill_len, update_end + local_len)`

`available_context = update_end` in both.

### 4.1 Page append

For each of the `M` tokens in order, with `s = owner[m]`:

```
lane = slot_lengths[s] % 16
if lane == 0:  p = next_page++;  slot_pages[s][slot_lengths[s]/16] = p
else:          p = slot_pages[s][slot_lengths[s]/16]
page_indices[p][lane] = absolute position of token m
page_sum_k[p] += k_m ;  page_sum_v[p] += v_m ;  page_counts[p] += 1
slot_lengths[s] += 1
```

Pages are therefore **per slot**, filled in arrival order, and a slot's last
page is partial.  `sum over pages of page_counts == count[s]`.

---

## 5. Attention

Let `R = prefill_two_level_topk` (3) when `n_tokens > 1`, else `two_level_topk`
(8).  `scale = 1/sqrt(D)` (the model's own scaling).

### 5.1 Routing

```
logit[q][s] = scale * dot(q, mean_k_s) + log(max(count_s,1))
logit[q][s] = -inf where count_s <= 0.5 or s < sink_len
top_slots[q][0..R-1] = indices of the R largest logits (descending)
```

Routing is **per query token and per query head** (GQA heads share the KV head's
state but route independently).  A route entry of -1 means "not opened".

### 5.2 Coarse branch

Softmax over all `state_len` slots with `logit` as above, except every slot in
`top_slots[q]` is forced to `-inf`.  Values are `mean_v_s`.  Returns
`(out_coarse, lse_coarse)`.

### 5.3 Local branch

Exact causal attention of the query block against KV positions
`[bswa_begin, query_end)`.  Returns `(out_local, lse_local)`.

### 5.4 Routed branch (recursive one page per region)

For each opened slot `s` of query `q`:

```
# pick the page
for each page p owned by s:
    pscore[p] = scale * dot(q, page_sum_k[p]/page_counts[p]) + log(page_counts[p])
p* = argmax pscore

# exact tokens of that page: 16 logits
tok_logit[i] = scale * dot(q, K[page_indices[p*][i]])   , -inf if lane empty

# one residual entry standing for the rest of the slot
rc = count_s - page_counts[p*]
res_k = (state_k[s] - page_sum_k[p*]) / max(rc,1)
res_v = (state_v[s] - page_sum_v[p*]) / max(rc,1)
res_logit = scale * dot(q, res_k) + log(max(rc,1))     , -inf if rc <= 0
```

The routed branch softmaxes over the concatenation of all opened slots'
`(residual, 16 tokens)` entries, i.e. `R * 17` logits, with values
`(res_v, V[page_indices[p*][i]])`.  Returns `(out_exact, lse_exact)`.

If no route is open, `out_exact = 0` and `lse_exact = -inf`.

### 5.5 Merge

```
w = softmax([lse_coarse, lse_local, lse_exact])
out = w0*out_coarse + w1*out_local + w2*out_exact
```

---

## 6. Schedule

### 6.1 Prefill

```
exact_lookback = prefill_local_len - prefill_chunk_len          # 768
front_len      = exact_lookback + chunk_len                     # 1024
```

* tokens `[0, front_len)`: plain exact causal attention, no LoD at all
* initial state: the first `chunk_len` tokens become singleton slots;
  `coverage = chunk_len`
* for `query_begin` in `front_len, +prefill_chunk_len ...`:
  * `bswa_begin = max(0, query_begin - exact_lookback)`;
    **assert `coverage == bswa_begin`**
  * three-branch attention for `[query_begin, query_end)` with local field
    `[bswa_begin, query_end)`
  * advance `coverage` to `max(0, query_begin + prefill_chunk_len -
    exact_lookback)` in steps of `prefill_state_update_len`
* after the loop, advance `coverage` to `max(chunk_len, bswa_begin(N+1))` so
  the first decoded token does not pay a full state update

### 6.2 Decode

```
bswa_begin(T) = max(0, round_up(T, chunk_len) - local_len)
```

Per token: append K/V to the archive, advance `coverage` to `bswa_begin(T+1)`
in steps of `decode_state_update_len`, then three-branch attention with
`R = 8` and local field `[coverage, T]`.

### 6.3 Mapping onto llama.cpp

`prefill_chunk_len` becomes the **ubatch size** (`-ub`).  With `-ub 4096` the
schedule is exactly the reference's.  Everything else is a function of the
ubatch boundary, so the graph shapes stay static:

```
for a ubatch covering positions [p0, p1):
    bswa_begin      = max(0, p0 - exact_lookback)
    next_bswa_begin = max(0, p1 - exact_lookback)
    if bswa_begin == 0: plain causal attention over [0, p1)
    else:               three branches, local field [bswa_begin, p1)
    then advance coverage from bswa_begin to next_bswa_begin
```

`state_len` and every loop bound are functions of positions only, so they are
known at graph-build time.  Only the *contents* (which token appends, which
slot it merges into, which page wins) are data dependent.

---

## 7. Numerics and known behaviours

* **The kernel engine is nondeterministic on purpose.**  Page summaries are
  accumulated with `atomic_add`/`scatter_add_`, and the downstream `argmax`
  over pages turns a rounding difference into a different answer.  Single-run
  scores are samples, not measurements.  The port should aim for a
  *deterministic* accumulation order; that is an improvement, not a deviation.
* INT4 leaves: one BF16 mean anchor per page plus one BF16 symmetric residual
  scale per 32 channels; 4-bit residual, values in `[-7,7]` stored as `+8` in
  a nibble.  Measured lossless against BF16 on 32k NIAH.
* Page summaries are stored INT8 in the kernel path (`page_summary_quant_bits`).
* Sums-based state means a suffix cannot be removed by rewinding a position.
  Prefix reuse, context shift and speculative rollback all need explicit
  support; phase 1 must refuse them rather than silently corrupt the state.
* `state_growth_factor=32` recovers multi-key documents at 32k that 16 loses,
  at about 70 MB of extra state - cheap next to the leaf archive.

---

## 8. llama.cpp op and storage design

### 8.1 Persistent tensors (per layer, per stream, allocated with the KV cache)

Packed so that each op stays inside `GGML_MAX_SRC = 10`.

```
s_kv   F32 [D + Dv + 1, S_cap, H_kv]   slot key_sum | value_sum | count
p_kv   F32 [D + Dv + 1, P_cap, H_kv]   page key_sum | value_sum | count
p_idx  I32 [ps,         P_cap, H_kv]   page lane -> KV cache cell id, -1 empty
s_pg   I32 [PPS + 1,    S_cap, H_kv]   slot page list | slot_length at [PPS]
meta   I32 [4,          H_kv]          next_page, reserved
```

The `+1` lanes let a single tensor carry both a vector and its scalar, so
`count` is a `ggml_view` with `ne0 = 1, nb1 = (D+Dv+1)*4`.  Nothing else needs
a separate allocation: the leaf archive is the KV cache.

For Qwen3.5-2B at 32k (`D = Dv = 256`, `H_kv = 2`, `S_cap = 3072`,
`P_cap = 5120`, `PPS = 128`) this is about 13 MB per layer, 78 MB for the six
full-attention layers.

### 8.2 Ops

Both follow the `ggml_set_rows` precedent: the result is
`ggml_view_tensor(ctx, <mutated tensor>)`, so the write lands in the
externally allocated persistent buffer and downstream nodes get a real data
dependency.

```
ggml_lod2_update(ctx, k_cache, v_cache, s_kv, p_kv, p_idx, s_pg, meta,
                 begin, end, state_len_in, state_len_out, sink_len, ps, pps)
    -> view of s_kv
```

One KV-means step (spec 4) followed by the page append (spec 4.1) for archive
positions `[begin, end)`.  Every bound is a host-side scalar known at graph
build time.

```
ggml_lod2_attn(ctx, q, k_cache, v_cache, s_kv, p_kv, p_idx, s_pg,
               local_begin, q_begin, state_len, routes, sink_len, ps, pps,
               scale)
    -> [Dv, H_q, T] F32
```

Routing, the three branches and the LSE merge (spec 5).

### 8.3 Graph shape

Per ubatch `[p0, p1)` on a LoD2 layer:

```
K/V writes into the cache                      (existing)
for each state block in [max(0,p0-lb), max(0,p1-lb)):
    ggml_lod2_update(...)                      (0..ceil(ub/1280) nodes)
ggml_lod2_attn(...)                            (1 node)
```

The number of update nodes and every parameter depend only on `p0`, `p1` and
the config, so graph reuse works exactly as it does for the dense path.

### 8.4 The invariant, and the one structural difference

**Invariant.** The exact window starts exactly where the state ends.  The graph
does not assert that the state reached a planned target; it advances the state
and then uses `coverage` itself as the window start.  Any other boundary would
drop tokens or count them twice, and this is the only place where that could go
wrong silently.

So each ubatch first absorbs everything it will not read exactly

* prefill (`n_tokens > 1`): up to `local_begin(p0)`, in `prefill_state_update`
  steps, budget measured at `end + lookback`
* generation: up to `decode_begin(p1)`, in `decode_state_update` steps, budget
  measured at `min(total, end + local_len)`

and then attends with `l0 = coverage`.  That is the reference's own ordering.

**Structural difference.** The reference's LoD query blocks start at
`front_len` (1024) and step by `prefill_chunk`; in llama.cpp they are the
ubatch grid, so they start at 0 and step by `n_ubatch`.  The consequence is
that the **first ubatch is attended exactly** (its window start is
`local_begin(0) = 0`) instead of only its first 1024 tokens.  Every later
ubatch matches the reference's structure.  This is more exact and costs one
quadratic ubatch that the dense path pays anyway.

Because of this, a bit-comparable oracle must be generated on llama.cpp's grid;
`lod2_ref.py` takes the grid as configuration precisely so that is possible.

### 8.5 Phase 1 restrictions (asserted, not silently ignored)

* one sequence (`n_seq_max == 1`), contiguous positions, no padding
* `f16`/`f32` KV cache only (the INT4 leaf format is phase 2)
* no context shift, no prefix reuse, no speculative rollback: the state is a
  sum, so a rewind needs explicit undo

## 9. Port plan

| phase | content | state |
|---|---|---|
| 0 | reference runs here; Qwen3.5-2B GGUF; dense parity | **done** |
| 0b | standalone PyTorch restatement + oracle cases | **done** |
| 1a | `ggml_lod2_update` / `ggml_lod2_attn`, CPU implementation, replay test | **done** |
| 1b | KV cache storage, graph builder, `LLAMA_LOD2` | **done (CPU)** |
| 1c | HIP/CUDA implementation of both ops | **done (correctness)** |
| 2 | fuse, static shapes, multi-GPU pipeline safety | todo |

### What is verified, and how

```
author's LOD attention  ==  docs/lod2-pytorch/lod2_ref.py  ==  ggml_lod2_* (CPU)
        |                              |                            |
   check_lod2_ref.py            dump_lod2_case.py             tests/test-lod2.cpp
```

* `check_lod2_ref.py` feeds identical post-RoPE q/k/v to the author's
  `PagedTwoLevelLODAttention` and to the restatement.  At 1k and 4k, 8q/2kv,
  the state (`key_sum`, `value_sum`, `count`) is **bit identical** and the
  attention output differs by 1.3e-6 to 1.7e-6, i.e. float accumulation order.
* `check_lod2_ref.py --dense-check` verifies the refinement property: with the
  state budget raised so every token is its own slot and every slot opened,
  LoD2 reproduces dense causal attention (5.7e-7).  This is what pins down the
  partition, the page index, the coarse exclusion and the LSE merge at once.
* `tests/test-lod2.cpp` replays an oracle case through the real ops using the
  schedule from `src/llama-lod2.h`.  Three cases pass: 1k/4q/64,
  2k/8q/128 (torch schedule), and 6k/8q/128 on the **paper's kernel schedule**
  (chunk 4096, local 4864, update 1280, 3 routes).  Counts are exact, outputs
  agree to 2.1e-6.

* End to end in llama.cpp, Qwen3.5-2B BF16, 1500-token prompt, `-ub 512`:
  with the state budget raised so every column is its own slot and every slot
  opened (`LLAMA_LOD2_GROWTH=1000000 LLAMA_LOD2_ROUTES=100000
  LLAMA_LOD2_ROUTES_PREFILL=100000`), 32 greedy tokens are **identical** to the
  dense path.  That exercises the KV cache storage, the graph schedule, the
  state recurrence and the merge against the real model.

Regenerate the cases with:

```sh
python docs/lod2-pytorch/dump_lod2_case.py --out /tmp/lod2-1k.bin \
    --len 1024 --decode 4 --query-heads 4 --kv-heads 2 --head-dim 64 --device cpu
./build/bin/test-lod2 /tmp/lod2-1k.bin
```

Constraints carried over from the LoD1 work:

* no model-specific code (no "Qwen special")
* ASCII only in files
* all GPU runs through `./simple-run.sh`
* no commits or pushes without explicit approval
