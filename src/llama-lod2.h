#pragma once

// LoD2 schedule (ref. "Key-Value Means", arXiv:2605.09877; docs/lod2-port-spec.md).
//
// Everything here is a function of positions only, so the graph shapes and
// every op parameter are known when the graph is built.  The header is
// dependency-free on purpose: the graph builder and the standalone op test both
// include it, so there is exactly one definition of the schedule.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct llama_lod2_params {
    // state and window geometry
    uint32_t chunk_len   = 256;
    uint32_t local_len   = 512;
    uint32_t sink_len    = 1;
    uint32_t page_size   = 16;
    uint32_t pages_per_slot = 128;

    float    state_growth = 16.0f;
    uint32_t state_min    = 256;

    // routes
    uint32_t routes_prefill = 3;
    uint32_t routes_decode  = 8;

    // prefill schedule; prefill_chunk is the ubatch size in llama.cpp
    uint32_t prefill_chunk        = 4096;
    uint32_t prefill_local        = 4864;
    uint32_t prefill_state_update = 1280;
    uint32_t decode_state_update  = 256;

    uint32_t lookback() const {
        return prefill_local > prefill_chunk ? prefill_local - prefill_chunk : 0;
    }

    // How many update blocks a prefill ubatch may need.  The plan advances
    // coverage by at most one ubatch per call in steps of prefill_state_update,
    // and the very first block of a sequence is chunk_len wide instead, so one
    // spare covers the seed.  The graph pads to this count: a ggml graph whose
    // node count changes between ubatches forces ggml_gallocr to reallocate,
    // and that path synchronizes every backend, which drains the scheduler's
    // pipeline.  See docs/lod2-port-handover.md section 4h.
    uint32_t prefill_blocks_max(uint32_t n_ubatch) const {
        const uint32_t step = prefill_state_update > 0 ? prefill_state_update : 1;
        return 1 + (n_ubatch + step - 1)/step;
    }

    // tokens attended exactly, with no state at all, at the start of a sequence
    uint32_t front_len() const { return lookback() + chunk_len; }

    // where the exact local window starts for a query at position p
    uint32_t local_begin(uint32_t p) const {
        return p > lookback() ? p - lookback() : 0;
    }

    // decode window start once the sequence holds n tokens
    uint32_t decode_begin(uint32_t n) const {
        const uint32_t end = (n + chunk_len - 1)/chunk_len*chunk_len;
        return end > local_len ? end - local_len : 0;
    }

    uint32_t desired_slots(uint32_t ctx_len, uint32_t avail, uint32_t cur) const {
        const uint32_t target = std::max<uint32_t>(
                (uint32_t) std::floor(state_growth*std::sqrt((double) ctx_len)), state_min);
        return std::max(cur, std::min(target, avail));
    }

    // slot capacity to allocate for a context of n_ctx tokens
    uint32_t slot_capacity(uint32_t n_ctx) const {
        const uint32_t t = desired_slots(n_ctx + chunk_len, n_ctx + chunk_len, 0);
        return (t + chunk_len - 1)/chunk_len*chunk_len;
    }

    // page-pool capacity: one page per 'page_size' tokens plus one partial page
    // per slot
    uint32_t page_capacity(uint32_t n_ctx) const {
        const uint32_t seq = (n_ctx + chunk_len - 1)/chunk_len*chunk_len + chunk_len;
        return (seq + page_size - 1)/page_size + slot_capacity(n_ctx);
    }
};

// one ggml_lod2_update node
struct llama_lod2_block {
    uint32_t p0;        // first archive column absorbed
    uint32_t p1;        // one past the last
    uint32_t s_len;     // live slots before
    uint32_t s_len_new; // live slots after
};

// Advance the state from 'coverage' to 'target'.  'ctx_at(end)' is the context
// length the growth budget is measured against; the two author implementations
// differ only in this function (spec 4).
template <typename CtxFn>
inline void llama_lod2_plan(
        const llama_lod2_params & p,
        uint32_t coverage,
        uint32_t target,
        uint32_t s_len,
        uint32_t step,
        CtxFn ctx_at,
        std::vector<llama_lod2_block> & out) {
    while (coverage < target) {
        // the very first block seeds one slot per token, and is one chunk long
        const uint32_t width = s_len == 0 ? p.chunk_len : step;
        const uint32_t end   = std::min(target, coverage + width);

        const uint32_t s_new = s_len == 0
            ? end - coverage
            : p.desired_slots(ctx_at(end), end, s_len);

        out.push_back({ coverage, end, s_len, s_new });

        s_len    = s_new;
        coverage = end;
    }
}

// Prefill: the state must cover exactly [0, local_begin(p0)) before the
// attention for a ubatch starting at p0, and is then advanced to
// local_begin(p1) so the next ubatch finds it in place.
inline std::vector<llama_lod2_block> llama_lod2_plan_prefill(
        const llama_lod2_params & p,
        uint32_t coverage,
        uint32_t s_len,
        uint32_t target) {
    std::vector<llama_lod2_block> out;
    const uint32_t lb = p.lookback();
    llama_lod2_plan(p, coverage, target, s_len, p.prefill_state_update,
            [lb](uint32_t end) { return end + lb; }, out);
    return out;
}

// After the prompt, and during decode, the budget is measured against the
// context the state will actually be read at.
inline std::vector<llama_lod2_block> llama_lod2_plan_tail(
        const llama_lod2_params & p,
        uint32_t coverage,
        uint32_t s_len,
        uint32_t target,
        uint32_t total,
        uint32_t step) {
    std::vector<llama_lod2_block> out;
    const uint32_t local = p.local_len;
    llama_lod2_plan(p, coverage, target, s_len, step,
            [local, total](uint32_t end) { return std::min(total, end + local); }, out);
    return out;
}
