# lod_attention_ref.py - naive PyTorch reference of the llama.cpp LoD attention port.
#
# Mirrors the implementation specified in lod-attention-spec.md exactly (all-pages
# branch: page -> leaf, retained dense KV, layer/KV-head-shared selection). The
# function lod_attn() matches GGML_OP_LOD_ATTN's contract 1:1 so a compiled binding
# to ggml/src/ggml-cuda/lod-attn.cu can later be dropped in behind the same API.
#
# Verification order (see spec section 10):
#   1. full expansion (n_top >= P) equals dense to float tolerance  -> self_test()
#   2. PPL parity at n_top=32 on a real model
#   3. multikey3 needle (see lod-needle-mk3.sh): 6k/top16 must be 3/3
#   4. reproduce the OPEN mk3 mystery at 32k+ (see lod-attention-handover.md)

import math
import torch


# ---------------------------------------------------------------- state machine

class LoDCache:
    """Dense KV cache + derived page-sum index for ONE layer, one sequence.

    Invariants (spec sections 1 and 6):
      - cache rows are append-only, row index == token position
      - Ks/Vs hold raw (pre-quantization, post-RoPE) sums of exactly the tokens
        in [0, sums_pos); rewound rows are physically zeroed (RMW folds!)
    """

    def __init__(self, n_ctx, n_head_kv, d_head, page_size=64, dtype=torch.float32, device="cpu"):
        self.ps = page_size
        self.P_cap = n_ctx // page_size
        self.K = torch.zeros(n_ctx, n_head_kv, d_head, dtype=dtype, device=device)
        self.V = torch.zeros(n_ctx, n_head_kv, d_head, dtype=dtype, device=device)
        self.Ks = torch.zeros(self.P_cap, n_head_kv, d_head, dtype=torch.float32, device=device)
        self.Vs = torch.zeros(self.P_cap, n_head_kv, d_head, dtype=torch.float32, device=device)
        self.T = 0         # tokens stored
        self.sums_pos = 0  # tokens folded into Ks/Vs

    def append(self, k, v):
        """k, v: [n, Hkv, D] post-RoPE. Stores AND folds (like the fused kernel /
        prefill update_sums)."""
        n = k.shape[0]
        self.K[self.T:self.T + n] = k
        self.V[self.T:self.T + n] = v
        self.T += n
        self.catch_up()

    def catch_up(self):
        """Fold [sums_pos, T) from the cache (spec 6: rewound rows are zero)."""
        for t in range(self.sums_pos, self.T):
            p = t // self.ps
            self.Ks[p] += self.K[t].float()
            self.Vs[p] += self.V[t].float()
        self.sums_pos = self.T

    def truncate(self, p0):
        """Suffix removal at position p0 (prompt reuse / speculative rollback).
        Page-floor rewind + physical zeroing - the fix for the cached-prefill
        corruption bug (handover doc, 'stale sums')."""
        if p0 >= self.sums_pos:
            self.T = p0
            return
        lo = p0 // self.ps
        hi = (self.sums_pos + self.ps - 1) // self.ps
        self.Ks[lo:hi] = 0
        self.Vs[lo:hi] = 0
        self.sums_pos = lo * self.ps
        self.T = p0


# ---------------------------------------------------------------- the fused op

def lod_attn(q, K, V, Ks, Vs, sel, state, page_size, scale, n_top=0, sel_head=False):
    """Mirror of GGML_OP_LOD_ATTN (see ggml.h and lod-attention-spec.md section 3).

      q     [nq, Hq, Dk]   queries at positions state[0] .. state[0]+nq-1
      K, V  [T, Hkv, D]    dense cache (rows past state[0]+nq never read)
      Ks,Vs [P_cap, Hkv, D] raw page sums
      sel   LongTensor [n_sel] or [Hkv, n_sel] page ids, -1 padded; None = in-op
            selection of n_top pages (max over queries and (group-)heads, raw
            sums scores, ties -> lower page id)
      state (prev_end, P)  P = complete pages to read
      returns [nq, Hq, Dv]

    Folding is NOT done here (unlike the CUDA op) - call cache.append()/catch_up()
    first; this keeps the reference purely functional.
    """
    nq, Hq, Dk = q.shape
    Hkv = K.shape[1]
    g = Hq // Hkv
    prev_end, P = state
    ps = page_size
    tail0 = P * ps
    logn = math.log(ps)

    qf = q.float()
    # score[qi, h, p] = q[qi, h] . Ks[p, kv(h)]  (raw sums - ranking is scale-invariant)
    sc = torch.stack([
        torch.einsum("qd,pd->qp", qf[:, h], Ks[:P, h // g].float()) for h in range(Hq)
    ], dim=1)  # [nq, Hq, P]

    if sel is None:
        assert n_top > 0
        if sel_head:
            sel_rows = []
            for kv in range(Hkv):
                pooled = sc[:, kv * g:(kv + 1) * g].amax(dim=(0, 1))  # [P]
                sel_rows.append(_topk_lowtie(pooled, n_top))
            sel = torch.stack(sel_rows)                               # [Hkv, n_top]
        else:
            pooled = sc.amax(dim=(0, 1))                              # [P]
            sel = _topk_lowtie(pooled, n_top).unsqueeze(0)            # [1, n_top]
    elif sel.dim() == 1:
        sel = sel.unsqueeze(0)

    Dv = V.shape[2]
    out = torch.zeros(nq, Hq, Dv, dtype=torch.float32, device=q.device)
    for h in range(Hq):
        kv = h // g
        srow = sel[kv if sel.shape[0] > 1 else 0]
        pages = [int(p) for p in srow.tolist() if p >= 0]
        in_sel = torch.zeros(P, dtype=torch.bool, device=q.device)
        if pages:
            in_sel[pages] = True
        for qi in range(nq):
            limit = prev_end + qi  # inclusive
            logits, values = [], []
            for p in pages:                                    # tier L: leaves
                ks_ = K[p * ps:(p + 1) * ps, kv].float()
                logits.append(scale * (ks_ @ qf[qi, h]))
                values.append(V[p * ps:(p + 1) * ps, kv].float())
            ke = K[tail0:limit + 1, kv].float()                # tier E: exact tail
            logits.append(scale * (ke @ qf[qi, h]))
            values.append(V[tail0:limit + 1, kv].float())
            uns = (~in_sel).nonzero().squeeze(-1)              # tier S: summaries
            if uns.numel():
                logits.append(scale * (Ks[uns, kv].float() / ps) @ qf[qi, h] + logn)
                values.append(Vs[uns, kv].float() / ps)
            lg = torch.cat(logits)
            w = torch.softmax(lg, dim=-1)                      # ONE shared softmax
            out[qi, h] = w @ torch.cat(values, dim=0)
    return out


def _topk_lowtie(scores, k):
    """top-k, strict >, ties -> lower page id; -1 padded (matches CPU/CUDA)."""
    P = scores.shape[0]
    k_eff = min(k, P)
    order = sorted(range(P), key=lambda p: (-float(scores[p]), p))
    sel = order[:k_eff] + [-1] * (k - k_eff)
    return torch.tensor(sel, dtype=torch.long, device=scores.device)


# --------------------------------------------------------------- prefill/decode

def prefill_segment(cache, q_seg, k_seg, v_seg, scale, n_top, sel_head=False):
    """One chunked-prefill segment (spec section 4): fold first, ONE shared
    selection pooled over the segment's queries, causal tail per query."""
    prev_end = cache.T
    cache.append(k_seg, v_seg)
    P = prev_end // cache.ps
    return lod_attn(q_seg, cache.K, cache.V, cache.Ks, cache.Vs, None,
                    (prev_end, P), cache.ps, scale, n_top, sel_head)


def decode_step(cache, q1, k1, v1, scale, n_top, sel_head=False):
    prev_end = cache.T
    cache.append(k1, v1)
    P = prev_end // cache.ps
    return lod_attn(q1, cache.K, cache.V, cache.Ks, cache.Vs, None,
                    (prev_end, P), cache.ps, scale, n_top, sel_head)


# ------------------------------------------------------------------- self test

def self_test():
    torch.manual_seed(0)
    n_ctx, Hkv, g, D, ps = 1024, 2, 4, 64, 16
    Hq = Hkv * g
    scale = 1.0 / math.sqrt(D)
    cache = LoDCache(n_ctx, Hkv, D, ps)
    T0 = 9 * ps + 5
    cache.append(torch.randn(T0, Hkv, D), torch.randn(T0, Hkv, D))

    q = torch.randn(1, Hq, D)
    P = cache.T // ps

    full = lod_attn(q, cache.K, cache.V, cache.Ks, cache.Vs, None,
                    (cache.T, P), ps, scale, n_top=P)
    dense = torch.stack([
        torch.stack([
            torch.softmax(scale * (cache.K[:cache.T + 1, h // g].float() @ q[0, h].float()), -1)
            @ cache.V[:cache.T + 1, h // g].float() for h in range(Hq)])])
    err = (full - dense).abs().max().item()
    print(f"full-expansion vs dense: max_err = {err:.3e}")
    assert err < 1e-5, "refinement invariant violated"

    part = lod_attn(q, cache.K, cache.V, cache.Ks, cache.Vs, None,
                    (cache.T, P), ps, scale, n_top=3)
    print(f"partial (top3) produced finite output: {torch.isfinite(part).all().item()}")

    cache.truncate(T0 - 7)  # mid-page suffix rollback
    cache.append(torch.randn(7, Hkv, D), torch.randn(7, Hkv, D))
    ref = torch.zeros_like(cache.Ks)
    for t in range(cache.T):
        ref[t // ps] += cache.K[t].float()
    err2 = (ref - cache.Ks).abs().max().item()
    print(f"truncate+refold sums err = {err2:.3e}")
    assert err2 < 1e-4, "stale-sums bug reproduced - truncate must zero rows"
    print("self_test OK")


if __name__ == "__main__":
    self_test()
