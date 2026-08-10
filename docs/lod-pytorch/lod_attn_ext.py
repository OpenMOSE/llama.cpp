# lod_attn_ext.py - JIT-compiled PyTorch binding for the extracted LoD kernels.
#
#   from lod_attn_ext import lod_attn_cuda
#   out = lod_attn_cuda(q, K, V, Ks, Vs, prev_end, P, scale, n_top, sel_head, ps)
#
# Same contract as lod_attention_ref.lod_attn() EXCEPT: like the ggml op, this
# FOLDS the current nq tokens into Ks/Vs in place (pass clones to keep them).
# Run this file directly for a parity test against the naive reference.

import math
import os

import torch
from torch.utils.cpp_extension import load

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lod_attn_kernels.cu")
        _ext = load(name="lod_attn_ext", sources=[src], verbose=os.environ.get("LOD_EXT_VERBOSE") == "1")
    return _ext


def lod_attn_cuda(q, K, V, Ks, Vs, prev_end, P, scale, n_top, sel_head=False, page_size=64):
    return _get_ext().lod_attn_forward(q, K, V, Ks, Vs, prev_end, P, float(scale),
                                       int(n_top), bool(sel_head), int(page_size))


# ------------------------------------------------------------------ parity test

def parity_test(device="cuda"):
    from lod_attention_ref import LoDCache, lod_attn

    torch.manual_seed(0)
    for (Hkv, g, D, ps, P, tail, nq, n_top, sel_head, f16) in [
        (4, 8, 512, 64, 8, 33, 1, 3, False, True),   # gemma-like decode
        (4, 8, 512, 64, 8, 33, 4, 8, False, True),   # small speculative batch
        (4, 6, 256, 64, 8, 33, 1, 3, False, True),   # qwen3.6-like
        (2, 2, 64, 64, 8, 33, 1, 10, False, False),  # f32 cache, n_top > P padding
        (4, 8, 512, 64, 8, 33, 1, 3, True, True),    # per-KV-head selection
    ]:
        Hq = Hkv*g
        T0 = P*ps + tail
        n_ctx = (P + 4)*ps
        scale = 1.0/math.sqrt(D)

        cache = LoDCache(n_ctx, Hkv, D, ps)
        cache.append(torch.randn(T0 + nq, Hkv, D), torch.randn(T0 + nq, Hkv, D))

        q = torch.randn(nq, Hq, D)

        # reference is pure (cache already folded); the CUDA op folds internally,
        # so hand it sums that exclude the current nq tokens
        Ks_pre = cache.Ks.clone()
        Vs_pre = cache.Vs.clone()
        for t in range(T0, T0 + nq):
            Ks_pre[t//ps] -= cache.K[t].float()
            Vs_pre[t//ps] -= cache.V[t].float()

        ref = lod_attn(q, cache.K, cache.V, cache.Ks, cache.Vs, None,
                       (T0, P), ps, scale, n_top, sel_head)

        dt = torch.float16 if f16 else torch.float32
        out = lod_attn_cuda(q.to(device), cache.K.to(device, dt), cache.V.to(device, dt),
                            Ks_pre.to(device), Vs_pre.to(device),
                            T0, P, scale, n_top, sel_head, ps)

        err = (out.cpu() - ref).abs().max().item()
        tol = 5e-3 if f16 else 1e-4  # f16 cache quantization dominates
        status = "OK " if err < tol else "FAIL"
        print(f"{status} Hkv={Hkv} g={g} D={D} nq={nq} top={n_top} head={int(sel_head)} "
              f"{'f16' if f16 else 'f32'}: max_err={err:.2e}")
        assert err < tol


if __name__ == "__main__":
    parity_test()
    print("parity OK - CUDA/HIP kernels match the naive reference")
