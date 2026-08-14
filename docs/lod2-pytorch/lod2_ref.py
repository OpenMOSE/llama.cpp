"""Standalone, deterministic PyTorch reference for LoD2 attention.

This file is the executable form of docs/lod2-port-spec.md.  It is written
from the author's implementation in /home/mose/Projects/KVM-paper/model/ but
it does not import it: the point is to have an independent restatement that
can be checked *against* the author's code, and that the llama.cpp port can be
checked against in turn.

Two schedules are supported through LoD2Config:

* ``paper_torch``  - what PagedTwoLevelLODAttention does
                     (chunk 256, local 512, routes 8 everywhere)
* ``paper_kernel`` - what KernelRecursivePagedLODAttention does
                     (prefill chunk 4096, prefill local 4864,
                      state update 1280, prefill routes 3, decode routes 8)

Only ``kv_bits=0`` (BF16/FP32 leaves) is implemented here.  INT4 leaf storage
is a storage format, not a different algorithm, and the author measured it as
lossless on 32k NIAH; it belongs in the llama.cpp cache layer, not in a
correctness oracle.

Layout follows the author's: q [B, QH, T, D], k/v [B, KVH, T, D], all
post-projection and post-RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


NEG_INF = float("-inf")


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass
class LoD2Config:
    """Every knob that changes a number, spelled out."""

    chunk_len: int = 256
    local_len: int = 512
    state_growth_factor: float = 16.0
    state_min_len: int = 256
    sink_len: int = 1
    page_size: int = 16
    inline_pages_per_slot: int = 128

    decode_routes: int = 8
    prefill_routes: int = 8

    prefill_chunk_len: int = 256
    prefill_local_len: int = 512
    prefill_state_update_len: int = 256
    decode_state_update_len: int = 256

    # The two author implementations disagree only on the ctx_len used by the
    # post-prefill updates that prepare the decode boundary:
    #   "total"          - PagedTwoLevelLODAttention passes the prompt length
    #   "end_plus_local" - the Triton engine passes min(N, end + local_len)
    # Everything before that boundary is identical in both.
    final_update_ctx: str = "total"

    @property
    def exact_lookback(self) -> int:
        return self.prefill_local_len - self.prefill_chunk_len

    @property
    def front_len(self) -> int:
        return self.exact_lookback + self.chunk_len

    def validate(self) -> None:
        if self.local_len % self.chunk_len:
            raise ValueError("local_len must be a multiple of chunk_len")
        if self.prefill_chunk_len > self.prefill_local_len:
            raise ValueError("prefill chunk cannot exceed its local field")
        if self.prefill_state_update_len <= 0:
            raise ValueError("prefill state update length must be positive")


def paper_torch_config() -> LoD2Config:
    return LoD2Config()


def paper_kernel_config() -> LoD2Config:
    chunk = 256
    return LoD2Config(
        chunk_len=chunk,
        local_len=512,
        prefill_routes=3,
        decode_routes=8,
        prefill_chunk_len=16 * chunk,
        prefill_local_len=16 * chunk + 512 + chunk,
        prefill_state_update_len=5 * chunk,
        decode_state_update_len=chunk,
        final_update_ctx="end_plus_local",
    )


# --------------------------------------------------------------------------
# state


@dataclass
class LoD2State:
    """Sums, counts and the per-slot page index.  One row per (batch, kv head)."""

    key_sum: torch.Tensor      # [B, H, S_cap, D]   f32
    value_sum: torch.Tensor    # [B, H, S_cap, Dv]  f32
    count: torch.Tensor        # [B, H, S_cap]      f32
    state_len: int

    slot_pages: torch.Tensor   # [B, H, S_cap, PPS] i64, -1 = empty
    slot_length: torch.Tensor  # [B, H, S_cap]      i64
    next_page: torch.Tensor    # [B, H]             i64

    page_index: torch.Tensor   # [B, H, P_cap, ps]  i64, -1 = empty
    page_sum_k: torch.Tensor   # [B, H, P_cap, D]   f32
    page_sum_v: torch.Tensor   # [B, H, P_cap, Dv]  f32
    page_count: torch.Tensor   # [B, H, P_cap]      f32

    coverage: int = 0
    total_len: int = 0

    @property
    def mean_key(self) -> torch.Tensor:
        return self.key_sum / self.count.clamp_min(1.0).unsqueeze(-1)

    @property
    def mean_value(self) -> torch.Tensor:
        return self.value_sum / self.count.clamp_min(1.0).unsqueeze(-1)


def desired_state_len(
    cfg: LoD2Config, ctx_len: int, available_context: int, current: int
) -> int:
    target = max(
        math.floor(cfg.state_growth_factor * math.sqrt(max(ctx_len, 0))),
        cfg.state_min_len,
    )
    return max(current, min(target, available_context))


def state_capacity(cfg: LoD2Config, total_len: int) -> int:
    target = desired_state_len(
        cfg, total_len + cfg.chunk_len, total_len + cfg.chunk_len, 0
    )
    return _round_up(target, cfg.chunk_len)


def new_state(
    cfg: LoD2Config,
    *,
    batch: int,
    kv_heads: int,
    head_dim: int,
    value_dim: int,
    total_len: int,
    device: torch.device,
) -> LoD2State:
    s_cap = state_capacity(cfg, total_len)
    seq_capacity = _round_up(total_len, cfg.chunk_len) + cfg.chunk_len
    p_cap = (seq_capacity + cfg.page_size - 1) // cfg.page_size + s_cap
    z = lambda *shape, dtype=torch.float32: torch.zeros(  # noqa: E731
        *shape, dtype=dtype, device=device
    )
    return LoD2State(
        key_sum=z(batch, kv_heads, s_cap, head_dim),
        value_sum=z(batch, kv_heads, s_cap, value_dim),
        count=z(batch, kv_heads, s_cap),
        state_len=0,
        slot_pages=torch.full(
            (batch, kv_heads, s_cap, cfg.inline_pages_per_slot),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        slot_length=z(batch, kv_heads, s_cap, dtype=torch.int64),
        next_page=z(batch, kv_heads, dtype=torch.int64),
        page_index=torch.full(
            (batch, kv_heads, p_cap, cfg.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        page_sum_k=z(batch, kv_heads, p_cap, head_dim),
        page_sum_v=z(batch, kv_heads, p_cap, value_dim),
        page_count=z(batch, kv_heads, p_cap),
    )


def _rank_within_owner(owner: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Arrival rank of each token among the tokens sharing its slot."""
    batch, heads, length = owner.shape
    order = torch.argsort(owner, dim=-1, stable=True)
    sorted_owner = owner.gather(-1, order)
    per_slot = torch.zeros(
        batch, heads, n_slots, dtype=torch.int64, device=owner.device
    )
    per_slot.scatter_add_(-1, owner, torch.ones_like(owner))
    starts = per_slot.cumsum(-1) - per_slot
    position = torch.arange(length, device=owner.device).view(1, 1, -1)
    rank_sorted = position - starts.gather(-1, sorted_owner)
    rank = torch.empty_like(owner)
    rank.scatter_(-1, order, rank_sorted)
    return rank


def append_pages(
    cfg: LoD2Config,
    state: LoD2State,
    key: torch.Tensor,     # [B, H, M, D]
    value: torch.Tensor,   # [B, H, M, Dv]
    owner: torch.Tensor,   # [B, H, M]
    positions: torch.Tensor,  # [M] absolute archive positions
) -> None:
    """Append one overflow block to the per-slot page index (see spec 4.1)."""
    batch, heads, length = owner.shape
    if length == 0:
        return
    ps = cfg.page_size
    n_slots = int(state.count.size(2))

    rank = _rank_within_owner(owner, n_slots)
    prior = state.slot_length.gather(-1, owner)
    index = prior + rank
    ordinal = torch.div(index, ps, rounding_mode="floor")
    lane = index.remainder(ps)

    added = torch.zeros(
        batch, heads, n_slots, dtype=torch.int64, device=owner.device
    )
    added.scatter_add_(-1, owner, torch.ones_like(owner))
    old_pages = torch.div(state.slot_length + ps - 1, ps, rounding_mode="floor")
    new_total = state.slot_length + added
    new_pages = (
        torch.div(new_total + ps - 1, ps, rounding_mode="floor") - old_pages
    )
    # Deterministic allocation: slots in ascending id, pages in ascending
    # ordinal.  The author's kernel uses an atomic bump allocator, so page ids
    # differ run to run there; ids are labels only, so this is a strengthening.
    page_base = state.next_page.unsqueeze(-1) + (new_pages.cumsum(-1) - new_pages)

    token_old_pages = old_pages.gather(-1, owner)
    token_base = page_base.gather(-1, owner)
    reuse = ordinal < token_old_pages
    fresh_id = token_base + (ordinal - token_old_pages)
    existing_id = state.slot_pages.gather(
        2, owner.unsqueeze(-1).expand(-1, -1, -1, cfg.inline_pages_per_slot)
    ).gather(-1, ordinal.clamp_max(cfg.inline_pages_per_slot - 1).unsqueeze(-1)
             ).squeeze(-1)
    page_id = torch.where(reuse, existing_id, fresh_id)

    if int(ordinal.max().item()) >= cfg.inline_pages_per_slot:
        raise RuntimeError(
            "slot posting list overflowed inline_pages_per_slot; the port must "
            "grow the table rather than drop pages"
        )

    # publish the freshly allocated ids into the per-slot posting list
    flat_slot = (
        owner * cfg.inline_pages_per_slot + ordinal
    ).reshape(batch, heads, -1)
    flat_pages = state.slot_pages.reshape(batch, heads, -1)
    flat_pages.scatter_(-1, flat_slot, page_id)

    # page contents
    p_cap = int(state.page_count.size(2))
    state.page_index.reshape(batch, heads, -1).scatter_(
        -1,
        (page_id * ps + lane).reshape(batch, heads, -1),
        positions.view(1, 1, -1).expand(batch, heads, -1),
    )
    state.page_sum_k.scatter_add_(
        2, page_id.unsqueeze(-1).expand_as(key), key.float()
    )
    state.page_sum_v.scatter_add_(
        2, page_id.unsqueeze(-1).expand_as(value), value.float()
    )
    state.page_count.scatter_add_(
        -1, page_id, torch.ones_like(page_id, dtype=torch.float32)
    )

    state.slot_length = new_total
    state.next_page = state.next_page + new_pages.sum(-1)
    if int(state.next_page.max().item()) > p_cap:
        raise RuntimeError("page pool overflowed")


def update_state(
    cfg: LoD2Config,
    state: LoD2State,
    key: torch.Tensor,     # [B, H, M, D]
    value: torch.Tensor,   # [B, H, M, Dv]
    positions: torch.Tensor,
    *,
    ctx_len: int,
    available_context: int,
) -> None:
    """One KV-means recurrence step (see spec 4)."""
    batch, heads, length, _ = key.shape
    if length == 0:
        return

    if state.state_len == 0:
        if length > int(state.count.size(2)):
            raise ValueError("initial block exceeds the state capacity")
        state.key_sum[..., :length, :] = key.float()
        state.value_sum[..., :length, :] = value.float()
        state.count[..., :length] = 1.0
        state.state_len = length
        owner = (
            torch.arange(length, device=key.device)
            .view(1, 1, -1)
            .expand(batch, heads, -1)
            .contiguous()
        )
        append_pages(cfg, state, key, value, owner, positions)
        return

    current = state.state_len
    desired = desired_state_len(cfg, ctx_len, available_context, current)
    n_append = min(max(desired - current, 0), length)
    protected = min(cfg.sink_len, current)

    mean_key = state.mean_key[..., :current, :]
    active = state.count[..., :current] > 0.5
    similarity = torch.matmul(key.float(), mean_key.transpose(-1, -2))
    similarity = similarity.masked_fill(~active.unsqueeze(-2), NEG_INF)

    if protected:
        protected_score = similarity[..., :protected].max(dim=-1).values
        similarity[..., :protected] = NEG_INF
    else:
        protected_score = torch.full_like(similarity[..., 0], NEG_INF)
    best_score, best_slot = similarity.max(dim=-1)
    select = torch.maximum(best_score, protected_score)

    order = torch.argsort(select, dim=-1, descending=False, stable=True)
    append_idx = torch.sort(order[..., :n_append], dim=-1).values
    merge_idx = torch.sort(order[..., n_append:], dim=-1).values

    owner = torch.full(
        (batch, heads, length), -1, dtype=torch.int64, device=key.device
    )

    if n_append:
        gk = append_idx.unsqueeze(-1).expand(-1, -1, -1, key.size(-1))
        gv = append_idx.unsqueeze(-1).expand(-1, -1, -1, value.size(-1))
        append_key = key.gather(2, gk).float()
        append_value = value.gather(2, gv).float()
        state.key_sum[..., current:current + n_append, :] = append_key
        state.value_sum[..., current:current + n_append, :] = append_value
        state.count[..., current:current + n_append] = 1.0
        append_slot = (
            torch.arange(current, current + n_append, device=key.device)
            .view(1, 1, -1)
            .expand_as(append_idx)
        )
        owner.scatter_(-1, append_idx, append_slot)

    n_merge = int(merge_idx.size(-1))
    if n_merge:
        gk = merge_idx.unsqueeze(-1).expand(-1, -1, -1, key.size(-1))
        gv = merge_idx.unsqueeze(-1).expand(-1, -1, -1, value.size(-1))
        merge_key = key.gather(2, gk).float()
        merge_value = value.gather(2, gv).float()
        dest = best_slot.gather(-1, merge_idx)
        merge_best = best_score.gather(-1, merge_idx)
        if n_append:
            appended = torch.matmul(merge_key, append_key.transpose(-1, -2))
            appended_score, appended_rel = appended.max(dim=-1)
            use_new = appended_score > merge_best
            dest = torch.where(use_new, appended_rel + current, dest)
        assignment = torch.nn.functional.one_hot(
            dest, num_classes=desired
        ).to(merge_key.dtype)
        assignment_t = assignment.transpose(-1, -2)
        state.key_sum[..., :desired, :] += torch.matmul(assignment_t, merge_key)
        state.value_sum[..., :desired, :] += torch.matmul(
            assignment_t, merge_value
        )
        state.count[..., :desired] += assignment_t.sum(-1)
        owner.scatter_(-1, merge_idx, dest)

    state.state_len = desired
    if bool((owner < 0).any().item()):
        raise AssertionError("state update left a token unowned")
    append_pages(cfg, state, key, value, owner, positions)


# --------------------------------------------------------------------------
# attention


def _repeat_kv(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    kv_heads = int(tensor.size(1))
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by kv heads")
    groups = query_heads // kv_heads
    return tensor if groups == 1 else tensor.repeat_interleave(groups, dim=1)


def _gather_h(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather rows of a [B,KVH,X,D] table with [B,QH,N] ids -> [B,QH,N,D].

    Query head h reads KV head h//groups, matching repeat_interleave, without
    ever materializing the expanded [B,QH,X,D] copy.
    """
    batch, kv_heads, _, dim = tensor.shape
    query_heads, count = int(index.size(1)), int(index.size(2))
    groups = query_heads // kv_heads
    flat = index.reshape(batch, kv_heads, groups * count)
    out = tensor.gather(2, flat.unsqueeze(-1).expand(-1, -1, -1, dim))
    return out.reshape(batch, query_heads, count, dim)


def _gather_h1(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Same as _gather_h for a scalar-valued [B,KVH,X] table."""
    batch, kv_heads, _ = tensor.shape
    query_heads, count = int(index.size(1)), int(index.size(2))
    groups = query_heads // kv_heads
    flat = index.reshape(batch, kv_heads, groups * count)
    return tensor.gather(2, flat).reshape(batch, query_heads, count)


def _softmax_branch(
    scores: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax a score block and return (output, logsumexp).

    ``values`` is either shared across queries (``[..., N, Dv]``, one rank less
    than ``scores`` plus the value dim) or per query (``[..., Q, N, Dv]``).
    A row whose logits are all -inf contributes nothing and yields lse=-inf.
    """
    lse = torch.logsumexp(scores, dim=-1)
    finite = torch.isfinite(lse)
    safe = torch.where(finite.unsqueeze(-1), scores, torch.zeros_like(scores))
    probability = torch.softmax(safe, dim=-1)
    probability = torch.where(
        finite.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    if values.ndim == scores.ndim + 1:
        output = torch.matmul(probability.unsqueeze(-2), values).squeeze(-2)
    else:
        output = torch.matmul(probability, values)
    return output, lse


def route_and_coarse(
    cfg: LoD2Config,
    state: LoD2State,
    query: torch.Tensor,   # [B, QH, Q, D] f32
    *,
    routes: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routing and the coarse branch (spec 5.1, 5.2)."""
    query_heads = int(query.size(1))
    s_len = state.state_len
    mean_key = _repeat_kv(state.mean_key[..., :s_len, :], query_heads)
    mean_value = _repeat_kv(state.mean_value[..., :s_len, :], query_heads)
    count = _repeat_kv(state.count[..., :s_len].unsqueeze(-1), query_heads
                       ).squeeze(-1)

    logits = torch.matmul(query, mean_key.transpose(-1, -2)) * scale
    logits = logits + count.clamp_min(1.0).log().unsqueeze(2)
    logits = logits.masked_fill((count <= 0.5).unsqueeze(2), NEG_INF)

    protected = min(cfg.sink_len, s_len)
    route_count = min(routes, s_len - protected)
    if route_count > 0:
        route_logits = logits.clone()
        if protected:
            route_logits[..., :protected] = NEG_INF
        top_slots = route_logits.topk(route_count, dim=-1).indices
        opened = torch.zeros_like(logits, dtype=torch.bool)
        opened.scatter_(-1, top_slots, True)
        # a slot whose logit is -inf was never really a candidate
        opened &= torch.isfinite(logits)
        coarse_logits = logits.masked_fill(opened, NEG_INF)
        top_slots = torch.where(
            opened.gather(-1, top_slots), top_slots, torch.full_like(top_slots, -1)
        )
    else:
        top_slots = torch.empty(
            *query.shape[:3], 0, dtype=torch.int64, device=query.device
        )
        coarse_logits = logits

    coarse_out, coarse_lse = _softmax_branch(coarse_logits, mean_value)
    return top_slots, coarse_out, coarse_lse


def local_branch(
    query: torch.Tensor,    # [B, QH, Q, D]
    key: torch.Tensor,      # [B, KVH, L, D]
    value: torch.Tensor,    # [B, KVH, L, Dv]
    *,
    query_offset: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact causal attention over the local field (spec 5.3)."""
    query_heads = int(query.size(1))
    key = _repeat_kv(key, query_heads)
    value = _repeat_kv(value, query_heads)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    q_index = torch.arange(int(query.size(2)), device=query.device).unsqueeze(-1)
    k_index = torch.arange(int(key.size(2)), device=query.device).unsqueeze(0)
    scores = scores.masked_fill(k_index > q_index + query_offset, NEG_INF)
    return _softmax_branch(scores, value)


def routed_branch(
    cfg: LoD2Config,
    state: LoD2State,
    query: torch.Tensor,      # [B, QH, Q, D]
    top_slots: torch.Tensor,  # [B, QH, Q, R]
    leaf_key: torch.Tensor,   # [B, KVH, T, D]
    leaf_value: torch.Tensor, # [B, KVH, T, Dv]
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One exact page plus a count-corrected residual per route (spec 5.4)."""
    batch, query_heads, query_len, route_count = top_slots.shape
    value_dim = int(leaf_value.size(-1))
    if route_count == 0:
        return (
            query.new_zeros(batch, query_heads, query_len, value_dim),
            torch.full(
                (batch, query_heads, query_len), NEG_INF, device=query.device
            ),
        )
    ps = cfg.page_size
    valid_slot = top_slots >= 0
    slot = top_slots.clamp_min(0)

    flat_slot_pages = state.slot_pages.reshape(
        batch, int(state.slot_pages.size(1)), -1
    )

    slot_flat = slot.reshape(batch, query_heads, -1)
    n_pages = torch.div(
        _gather_h1(state.slot_length, slot_flat) + ps - 1, ps,
        rounding_mode="floor",
    ).reshape(slot.shape)
    max_pages = int(n_pages.max().item()) if n_pages.numel() else 0

    best_score = torch.full(slot.shape, NEG_INF, device=query.device)
    best_page = torch.zeros(slot.shape, dtype=torch.int64, device=query.device)
    for begin in range(0, max_pages, 16):
        width = min(16, max_pages - begin)
        offset = begin + torch.arange(width, device=query.device)
        ordinal = offset.view(1, 1, 1, 1, -1).expand(*slot.shape, width)
        ok = (ordinal < n_pages.unsqueeze(-1)) & valid_slot.unsqueeze(-1)
        flat = (
            slot.unsqueeze(-1) * cfg.inline_pages_per_slot + ordinal
        ).reshape(batch, query_heads, -1)
        page = _gather_h1(flat_slot_pages, flat)
        page = page.reshape(*slot.shape, width).clamp_min(0)
        pflat = page.reshape(batch, query_heads, -1)
        cnt = _gather_h1(state.page_count, pflat).reshape(*page.shape)
        ksum = _gather_h(state.page_sum_k, pflat).reshape(*page.shape, -1)
        score = (
            (query.unsqueeze(-2).unsqueeze(-2)
             * (ksum / cnt.clamp_min(1.0).unsqueeze(-1))).sum(-1) * scale
            + cnt.clamp_min(1.0).log()
        )
        score = score.masked_fill(~ok, NEG_INF)
        block_score, block_arg = score.max(dim=-1)
        chosen = page.gather(-1, block_arg.unsqueeze(-1)).squeeze(-1)
        better = block_score > best_score
        best_score = torch.where(better, block_score, best_score)
        best_page = torch.where(better, chosen, best_page)

    open_mask = valid_slot & torch.isfinite(best_score)
    pflat = best_page.reshape(batch, query_heads, -1)
    sel_count = _gather_h1(state.page_count, pflat).reshape(best_page.shape)
    sel_ksum = _gather_h(state.page_sum_k, pflat).reshape(*best_page.shape, -1)
    sel_vsum = _gather_h(state.page_sum_v, pflat).reshape(*best_page.shape, -1)
    sel_index = _gather_h(state.page_index, pflat).reshape(*best_page.shape, ps)

    lane = torch.arange(ps, device=query.device)
    token_ok = open_mask.unsqueeze(-1) & (lane < sel_count.unsqueeze(-1))
    token_ok &= sel_index >= 0
    safe_index = sel_index.clamp_min(0)
    iflat = safe_index.reshape(batch, query_heads, -1)
    tok_k = _gather_h(leaf_key, iflat).reshape(*safe_index.shape, -1)
    tok_v = _gather_h(leaf_value, iflat).reshape(*safe_index.shape, value_dim)
    tok_score = (
        query.unsqueeze(-2).unsqueeze(-2) * tok_k
    ).sum(-1) * scale
    tok_score = tok_score.masked_fill(~token_ok, NEG_INF)

    sflat = slot.reshape(batch, query_heads, -1)
    slot_count = _gather_h1(state.count, sflat).reshape(slot.shape)
    slot_ksum = _gather_h(state.key_sum, sflat).reshape(*slot.shape, -1)
    slot_vsum = _gather_h(state.value_sum, sflat).reshape(*slot.shape, value_dim)
    residual_count = slot_count - sel_count
    residual_ok = open_mask & (residual_count > 0)
    safe_rc = residual_count.clamp_min(1.0)
    residual_k = (slot_ksum - sel_ksum) / safe_rc.unsqueeze(-1)
    residual_v = (slot_vsum - sel_vsum) / safe_rc.unsqueeze(-1)
    residual_score = (
        (query.unsqueeze(-2) * residual_k).sum(-1) * scale + safe_rc.log()
    )
    residual_score = residual_score.masked_fill(~residual_ok, NEG_INF)

    scores = torch.cat((residual_score.unsqueeze(-1), tok_score), dim=-1)
    values = torch.cat((residual_v.unsqueeze(-2), tok_v), dim=-2)
    return _softmax_branch(scores.flatten(-2), values.flatten(-3, -2))


def merge_branches(
    parts: list[tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Exact logsumexp recombination of disjoint branches (spec 5.5)."""
    lse = torch.stack([p[1] for p in parts], dim=-1).float()
    weight = torch.softmax(lse, dim=-1)
    output = torch.zeros_like(parts[0][0])
    for index, (branch_out, _) in enumerate(parts):
        output = output + branch_out * weight[..., index].unsqueeze(-1)
    return output


def lod2_attention_block(
    cfg: LoD2Config,
    state: LoD2State,
    query: torch.Tensor,
    leaf_key: torch.Tensor,
    leaf_value: torch.Tensor,
    *,
    local_begin: int,
    query_begin: int,
    routes: int,
    scale: float,
    query_block: int = 256,
) -> torch.Tensor:
    """Three branches over one query block.

    The queries are processed in sub-blocks purely to bound the size of the
    intermediates; the result does not depend on ``query_block`` because every
    branch is computed independently per query.
    """
    outputs = []
    total = int(query.size(2))
    for begin in range(0, total, query_block):
        end = min(total, begin + query_block)
        sub = query[..., begin:end, :]
        sub_begin = query_begin + begin
        local_k = leaf_key[..., local_begin:sub_begin + (end - begin), :]
        local_v = leaf_value[..., local_begin:sub_begin + (end - begin), :]
        top_slots, coarse_out, coarse_lse = route_and_coarse(
            cfg, state, sub, routes=routes, scale=scale
        )
        local_out, local_lse = local_branch(
            sub, local_k, local_v,
            query_offset=sub_begin - local_begin, scale=scale,
        )
        exact_out, exact_lse = routed_branch(
            cfg, state, sub, top_slots, leaf_key, leaf_value, scale=scale
        )
        outputs.append(merge_branches([
            (coarse_out, coarse_lse),
            (local_out, local_lse),
            (exact_out, exact_lse),
        ]))
    return torch.cat(outputs, dim=2)


def _advance_state(
    cfg: LoD2Config,
    state: LoD2State,
    key: torch.Tensor,
    value: torch.Tensor,
    target: int,
    *,
    step: int,
    ctx_len_of,
) -> None:
    while state.coverage < target:
        end = min(target, state.coverage + step)
        if state.state_len == 0:
            end = min(target, state.coverage + cfg.chunk_len)
        positions = torch.arange(state.coverage, end, device=key.device)
        update_state(
            cfg,
            state,
            key[..., state.coverage:end, :],
            value[..., state.coverage:end, :],
            positions,
            ctx_len=ctx_len_of(end),
            available_context=end,
        )
        state.coverage = end


def lod2_prefill(
    cfg: LoD2Config,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, LoD2State]:
    """Full-sequence prefill following spec 6.1."""
    cfg.validate()
    batch, query_heads, total, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    query = query.float()
    key = key.float()
    value = value.float()

    state = new_state(
        cfg,
        batch=batch,
        kv_heads=int(key.size(1)),
        head_dim=head_dim,
        value_dim=int(value.size(-1)),
        total_len=total,
        device=query.device,
    )

    front = min(total, cfg.front_len)
    outputs = [
        local_branch(
            query[..., :front, :], key[..., :front, :], value[..., :front, :],
            query_offset=0, scale=scale,
        )[0]
    ]

    lookback = cfg.exact_lookback
    for query_begin in range(front, total, cfg.prefill_chunk_len):
        query_end = min(total, query_begin + cfg.prefill_chunk_len)
        bswa_begin = max(0, query_begin - lookback)
        _advance_state(
            cfg, state, key, value, bswa_begin,
            step=cfg.prefill_state_update_len,
            ctx_len_of=lambda end: end + lookback,
        )
        if state.coverage != bswa_begin:
            raise AssertionError("LoD2 prefill state coverage drifted")
        outputs.append(
            lod2_attention_block(
                cfg, state, query[..., query_begin:query_end, :], key, value,
                local_begin=bswa_begin, query_begin=query_begin,
                routes=cfg.prefill_routes, scale=scale,
            )
        )

    state.total_len = total
    decode_coverage = max(
        min(total, cfg.chunk_len), _bswa_begin(cfg, total + 1)
    )
    if cfg.final_update_ctx == "total":
        final_ctx = lambda end: total          # noqa: E731
    elif cfg.final_update_ctx == "end_plus_local":
        final_ctx = lambda end: min(total, end + cfg.local_len)  # noqa: E731
    else:
        raise ValueError(f"unknown final_update_ctx {cfg.final_update_ctx!r}")
    _advance_state(
        cfg, state, key, value, decode_coverage,
        step=cfg.prefill_state_update_len,
        ctx_len_of=final_ctx,
    )
    return torch.cat(outputs, dim=2), state


def _bswa_begin(cfg: LoD2Config, total_len: int) -> int:
    return max(0, _round_up(total_len, cfg.chunk_len) - cfg.local_len)


def lod2_decode_one(
    cfg: LoD2Config,
    state: LoD2State,
    query: torch.Tensor,   # [B, QH, 1, D]
    key: torch.Tensor,     # [B, KVH, T+1, D] full archive including the new token
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """One decode step following spec 6.2."""
    if scale is None:
        scale = 1.0 / math.sqrt(int(query.size(-1)))
    total = state.total_len + 1
    if int(key.size(2)) != total:
        raise ValueError("decode archive length does not match the state")
    query = query.float()
    key = key.float()
    value = value.float()
    _advance_state(
        cfg, state, key, value, _bswa_begin(cfg, total),
        step=cfg.decode_state_update_len,
        ctx_len_of=lambda end: min(total, end + cfg.local_len),
    )
    state.total_len = total
    return lod2_attention_block(
        cfg, state, query, key, value,
        local_begin=state.coverage, query_begin=total - 1,
        routes=cfg.decode_routes, scale=scale,
    )


__all__ = [
    "LoD2Config",
    "LoD2State",
    "append_pages",
    "lod2_attention_block",
    "lod2_decode_one",
    "lod2_prefill",
    "local_branch",
    "merge_branches",
    "new_state",
    "paper_kernel_config",
    "paper_torch_config",
    "route_and_coarse",
    "routed_branch",
    "update_state",
]
