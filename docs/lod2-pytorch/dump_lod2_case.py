"""Emit a LoD2 oracle case for the llama.cpp side to replay.

The C++ test loads one of these files, runs the ported ops over the same q/k/v
with the same configuration, and compares the attention outputs and the final
state.  Page ids are allocator labels and are deliberately not compared; the
state sums, the counts and the outputs pin the algorithm down.

  python dump_lod2_case.py --out /tmp/lod2-case-1k.bin --len 1024 --decode 4

Binary layout (little endian, C order, f32 unless stated):

  char[8]  "LOD2CASE"
  i32      version = 1
  i32      n_head_q, n_head_kv, head_dim, value_dim
  i32      n_prompt, n_decode
  f32      scale
  i32      chunk_len, local_len, sink_len, page_size, inline_pages_per_slot
  i32      prefill_routes, decode_routes
  i32      prefill_chunk_len, prefill_local_len
  i32      prefill_state_update_len, decode_state_update_len
  f32      state_growth_factor
  i32      state_min_len
  i32      s_cap, p_cap, state_len, coverage
  f32      q         [n_prompt+n_decode][n_head_q ][head_dim ]
  f32      k         [n_prompt+n_decode][n_head_kv][head_dim ]
  f32      v         [n_prompt+n_decode][n_head_kv][value_dim]
  f32      out       [n_prompt+n_decode][n_head_q ][value_dim]
  f32      key_sum   [n_head_kv][s_cap][head_dim ]
  f32      value_sum [n_head_kv][s_cap][value_dim]
  f32      count     [n_head_kv][s_cap]
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lod2_ref as ref  # noqa: E402


def write_f32(fh, tensor: torch.Tensor) -> None:
    fh.write(tensor.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--query-heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--schedule", default="torch", choices=["torch", "kernel"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final-ctx", default="end_plus_local",
                   choices=["total", "end_plus_local"],
                   help="ctx_len used by the post-prompt updates; the port uses "
                        "end_plus_local (the kernel engine form)")
    a = ap.parse_args()

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    total = a.len
    scale = 1.0 / math.sqrt(a.head_dim)

    q = torch.randn(1, a.query_heads, total + a.decode, a.head_dim, device=device)
    k = torch.randn(1, a.kv_heads, total + a.decode, a.head_dim, device=device)
    v = torch.randn(1, a.kv_heads, total + a.decode, a.head_dim, device=device)

    cfg = ref.paper_torch_config() if a.schedule == "torch" else ref.paper_kernel_config()
    cfg.final_update_ctx = a.final_ctx

    with torch.no_grad():
        out, state = ref.lod2_prefill(
            cfg, q[..., :total, :], k[..., :total, :], v[..., :total, :], scale=scale
        )
        steps = [out]
        for step in range(a.decode):
            t = total + step
            steps.append(ref.lod2_decode_one(
                cfg, state, q[..., t:t + 1, :],
                k[..., :t + 1, :], v[..., :t + 1, :], scale=scale,
            ))
        out = torch.cat(steps, dim=2)

    s_cap = int(state.count.size(2))
    p_cap = int(state.page_count.size(2))

    with open(a.out, "wb") as fh:
        fh.write(b"LOD2CASE")
        fh.write(struct.pack("<i", 1))
        fh.write(struct.pack("<4i", a.query_heads, a.kv_heads, a.head_dim, a.head_dim))
        fh.write(struct.pack("<2i", total, a.decode))
        fh.write(struct.pack("<f", scale))
        fh.write(struct.pack(
            "<5i", cfg.chunk_len, cfg.local_len, cfg.sink_len, cfg.page_size,
            cfg.inline_pages_per_slot))
        fh.write(struct.pack("<2i", cfg.prefill_routes, cfg.decode_routes))
        fh.write(struct.pack("<2i", cfg.prefill_chunk_len, cfg.prefill_local_len))
        fh.write(struct.pack(
            "<2i", cfg.prefill_state_update_len, cfg.decode_state_update_len))
        fh.write(struct.pack("<f", cfg.state_growth_factor))
        fh.write(struct.pack("<i", cfg.state_min_len))
        fh.write(struct.pack(
            "<4i", s_cap, p_cap, state.state_len, state.coverage))
        # [B=1, H, T, D] -> [T, H, D]
        write_f32(fh, q[0].transpose(0, 1))
        write_f32(fh, k[0].transpose(0, 1))
        write_f32(fh, v[0].transpose(0, 1))
        write_f32(fh, out[0].transpose(0, 1))
        write_f32(fh, state.key_sum[0])
        write_f32(fh, state.value_sum[0])
        write_f32(fh, state.count[0])

    size = os.path.getsize(a.out)
    print(f"wrote {a.out} ({size/1e6:.1f} MB): schedule={a.schedule} "
          f"prompt={total} decode={a.decode} "
          f"heads={a.query_heads}/{a.kv_heads} dim={a.head_dim} "
          f"s_cap={s_cap} p_cap={p_cap} state_len={state.state_len} "
          f"coverage={state.coverage}")


if __name__ == "__main__":
    main()
