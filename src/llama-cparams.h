#pragma once

#include "llama.h"
#include "llama-lod2.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;

    // LoD sparse attention for full-attention layers (env: LLAMA_LOD*)
    bool     lod;
    bool     lod_sel_head;   // top-k page set per KV head (else one set per layer)
    bool     lod_fused;      // use the fused GGML_OP_LOD_ATTN read for decode
    uint32_t lod_page_size;
    uint32_t lod_top_pages;                 // max over layers (gates, reuse checks)
    std::vector<uint32_t> lod_top_pages_l;   // per-layer leaf budgets; 0 = dense escape
    std::vector<uint32_t> lod_top_regions_l; // per-layer region budgets (selection tier)
    // decode reads with its own budget: the two phases have different levers. At decode the
    // selection is already per-query (one token = one query), so the only thing a bigger
    // budget buys is coverage - which is what long generations need. Prefill shares one page
    // set across the ubatch, so there granularity matters more than budget. Decode budget is
    // also the cheaper one to raise (token generation is weight-bandwidth bound).
    uint32_t lod_top_pages_dec;
    std::vector<uint32_t> lod_top_pages_dec_l;
    std::vector<uint32_t> lod_top_regions_dec_l;

    // LoD2 content-addressed attention (env: LLAMA_LOD2*).  Unrelated to the
    // LoD fields above: the two engines share nothing but the branch point in
    // the model graphs, and enabling both at once is rejected.
    bool              lod2;
    llama_lod2_params lod2_p;
    std::vector<bool> lod2_l; // per-layer enable

    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
