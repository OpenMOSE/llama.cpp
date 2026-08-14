"""Check docs/lod2-pytorch/lod2_ref.py against the author's implementation.

The port is only as trustworthy as this check.  We feed identical post-RoPE
q/k/v to

  * the author's PagedTwoLevelLODAttention (preset torch_reference), and
  * lod2_ref.py configured with the same schedule,

and compare prefill outputs, the state (sums and counts), and a few decode
steps.  Both run in float32 on the same device, so agreement should be at
float-accumulation level, not "roughly similar".

  python check_lod2_ref.py --len 4096 --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mose/Projects/KVM-paper")

import lod2_ref as ref  # noqa: E402


def report(name: str, got: torch.Tensor, want: torch.Tensor) -> float:
    got = got.float()
    want = want.float()
    if got.shape != want.shape:
        raise SystemExit(f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
    diff = (got - want).abs()
    scale = want.abs().max().clamp_min(1e-6)
    print(
        f"  {name:<28} max_abs {diff.max().item():.3e}  "
        f"mean_abs {diff.mean().item():.3e}  rel {(diff.max()/scale).item():.3e}"
    )
    return float(diff.max().item())


def dense_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    query_heads = int(q.size(1))
    key = ref._repeat_kv(k, query_heads)
    value = ref._repeat_kv(v, query_heads)
    scores = torch.matmul(q.float(), key.float().transpose(-1, -2)) * scale
    length = int(q.size(2))
    index = torch.arange(length, device=q.device)
    scores = scores.masked_fill(index.unsqueeze(0) > index.unsqueeze(-1), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value.float())


def dense_check(a) -> None:
    """LoD2 is refinement, not approximation: open everything and it IS dense.

    Force every overflow token into its own slot (state budget >= context) and
    open every slot.  Each slot then holds one token in one page with an empty
    residual, so the routed branch reads the raw K/V and the union of the three
    branches is the whole causal history, each position exactly once.
    """
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    total = a.len
    scale = 1.0 / math.sqrt(a.head_dim)
    q = torch.randn(1, a.query_heads, total, a.head_dim, device=device)
    k = torch.randn(1, a.kv_heads, total, a.head_dim, device=device)
    v = torch.randn(1, a.kv_heads, total, a.head_dim, device=device)

    cfg = ref.paper_torch_config()
    cfg.state_growth_factor = 1.0e6      # every token becomes its own slot
    cfg.prefill_routes = total           # open every slot
    cfg.decode_routes = total
    with torch.no_grad():
        got, _ = ref.lod2_prefill(cfg, q, k, v, scale=scale)
        want = dense_reference(q, k, v, scale)
    print(f"refinement check, {total} tokens, every slot open")
    worst = report("lod2 vs dense", got, want)
    if worst > 1e-4:
        raise SystemExit("MISMATCH: full expansion is not dense")
    print("OK: full expansion reproduces dense attention")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=4096)
    ap.add_argument("--query-heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense-check", action="store_true",
                    help="check the refinement property instead of the author diff")
    a = ap.parse_args()

    if a.dense_check:
        dense_check(a)
        return

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    total = a.len
    scale = 1.0 / math.sqrt(a.head_dim)

    q = torch.randn(1, a.query_heads, total + a.decode, a.head_dim, device=device)
    k = torch.randn(1, a.kv_heads, total + a.decode, a.head_dim, device=device)
    v = torch.randn(1, a.kv_heads, total + a.decode, a.head_dim, device=device)

    from model.pytorch_lod_attention_paged import (  # noqa: E402
        PagedLODConfig,
        PagedTwoLevelLODAttention,
    )

    author_cfg = PagedLODConfig(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16.0,
        state_min_size=256,
        protected_prefix=1,
        max_routes=8,
        page_size=16,
        kv_bits=0,
        leaf_dtype=torch.float32,
    )
    author = PagedTwoLevelLODAttention(author_cfg, default_open_count=8).to(device)

    with torch.no_grad():
        author_out, author_cache = author(
            q[..., :total, :], k[..., :total, :], v[..., :total, :],
            use_cache=True, scale=scale,
        )

    cfg = ref.paper_torch_config()
    with torch.no_grad():
        mine_out, mine_state = ref.lod2_prefill(
            cfg, q[..., :total, :], k[..., :total, :], v[..., :total, :],
            scale=scale,
        )

    print(f"prefill {total} tokens, {a.query_heads}q/{a.kv_heads}kv x {a.head_dim}")
    worst = report("prefill output", mine_out, author_out)

    print(f"state slots: mine {mine_state.state_len} "
          f"author {author_cache.state.slot_count}, "
          f"coverage mine {mine_state.coverage} author {author_cache.coverage}")
    n = min(mine_state.state_len, author_cache.state.slot_count)
    worst = max(worst, report(
        "state count", mine_state.count[..., :n], author_cache.state.count[..., :n]))
    worst = max(worst, report(
        "state key_sum",
        mine_state.key_sum[..., :n, :], author_cache.state.key_sum[..., :n, :]))
    worst = max(worst, report(
        "state value_sum",
        mine_state.value_sum[..., :n, :], author_cache.state.value_sum[..., :n, :]))

    cache = author_cache
    for step in range(a.decode):
        t = total + step
        with torch.no_grad():
            author_step, cache = author(
                q[..., t:t + 1, :], k[..., t:t + 1, :], v[..., t:t + 1, :],
                cache=cache, use_cache=True, scale=scale,
            )
            mine_step = ref.lod2_decode_one(
                cfg, mine_state, q[..., t:t + 1, :],
                k[..., :t + 1, :], v[..., :t + 1, :], scale=scale,
            )
        worst = max(worst, report(f"decode {step}", mine_step, author_step))

    print(f"\nworst max_abs = {worst:.3e}")
    if worst > 2e-3:
        raise SystemExit("MISMATCH: the restatement disagrees with the author's code")
    print("OK: restatement agrees with the author's implementation")


if __name__ == "__main__":
    main()
