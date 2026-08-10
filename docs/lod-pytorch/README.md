# LoD attention - PyTorch reproduction package

Self-contained handover for reproducing (and continuing) the llama.cpp LoD port
on the PyTorch side. Everything here runs without building llama.cpp.

## Files

| file | what |
| --- | --- |
| `lod_attention_ref.py` | naive PyTorch reference: `LoDCache` state machine, `lod_attn()` (1:1 contract with `GGML_OP_LOD_ATTN`), `prefill_segment`/`decode_step`, `self_test()` (full expansion == dense, truncate+refold exactness) |
| `lod_attn_kernels.cu` | the production HIP/CUDA decode kernels, extracted **verbatim** from `ggml/src/ggml-cuda/lod-attn.cu` (same bytes = same numerics), plus a torch launcher for the in-op-selection path |
| `lod_attn_ext.py` | JIT loader (`torch.utils.cpp_extension`) + `lod_attn_cuda()` wrapper + parity test vs the reference |

Companion docs one level up: `../lod-attention-spec.md` (the algorithm),
`../lod-attention-handover.md` (project state, open questions),
`../lod-attention.md` (llama.cpp usage and measured results).

## Quick start

    python lod_attention_ref.py     # pure-python self test (CPU, seconds)
    python lod_attn_ext.py          # compiles the extension (ROCm: hipify is
                                    # automatic), runs CUDA-vs-reference parity

Contract notes:
- `lod_attn_cuda` (like the ggml op) FOLDS the current `nq` tokens into
  `Ks`/`Vs` in place; the python reference is pure - the parity test shows how
  to reconcile (hand the op sums that exclude the current tokens).
- shapes: `q [nq,Hq,Dk] f32`, `K/V [T,Hkv,D] f16|f32`, `Ks/Vs [P_cap,Hkv,D]
  f32`, all contiguous; `Dk % 64 == 0`, `nq <= 8`, GQA group `<= 32`.
- selection: in-op, layer-shared by default, `sel_head=True` for per-KV-head
  sets. Ties break toward the lower page id; `-1` pads when fewer pages exist.
- f16 caches: expect ~1e-3 differences vs the f32 reference (storage rounding);
  f32 caches match to ~1e-5.

## What to reproduce first (suggested order)

1. `self_test` + parity (this package) - the math and kernels are wired right.
2. PPL parity at `n_top=32` on a real model (llama.cpp measured: gemma 91.50
   vs dense 91.92 at c=4096; qwen3.6 4.269 vs 4.298).
3. multikey3 retrieval, 6k tokens, top16 -> must be 3/3 (see
   `../../lod-needle-mk3.sh` for the llama.cpp-side harness; 3 keys at
   10/50/90% depth, distractor phrasing shared between keys).
4. THE OPEN QUESTION (`../lod-attention-handover.md` section 3): at 32k+ the
   model answers from PREFILL logits with narrative continuation instead of
   retrieving, even though (a) decode-time selection ranks the needle pages
   top-10 in nearly all layers, (b) full expansion == dense == correct.
   PyTorch is the ideal place to dissect this: run the same haystack through
   `prefill_segment` chunks, inspect the final segment's selection, the
   question tokens' attention distribution over tiers, and the logits - the
   naive reference makes every intermediate observable. Known-eliminated
   causes: selection ranks, kernel bugs, tail pooling, budget (<= 512 of 1560
   pages all fail; 1024 passes).

## Faithfulness caveats

- The reference implements the DECODE/read semantics exactly. The llama.cpp
  prefill additionally does: FA-assembly with a shared per-segment selection
  (spec section 4), capacity-padded static graphs, and (work in progress)
  a mask-direct variant - none of that changes the math, only the schedule.
- Sums accumulate post-RoPE keys (same as llama.cpp and the original LoD1
  master). If you experiment with pre-RoPE selection sums (one of the fix
  candidates once the 32k mystery is resolved), that is a DELIBERATE deviation
  - keep it switchable.
- Page means (`k_mean`/`v_mean`, mask-direct prefill) are not part of this
  package; the reference reads summaries from the f32 sums directly.
