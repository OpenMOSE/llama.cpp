#include "llama-graph.h"

#include <algorithm>

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_lod::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    const uint32_t n_tokens = ubatch->n_tokens;

    // the LoD read assumes cell index == position (single seq, append-only, no shift)
    GGML_ASSERT((uint32_t) ubatch->pos[0] == prev_end);

    if (lod_ones) {
        std::vector<float> ones(ps, 1.0f);
        ggml_backend_tensor_set(lod_ones, ones.data(), 0, ggml_nbytes(lod_ones));
    }

    if (lod_meta) {
        const int32_t meta[2] = { (int32_t) prev_end, (int32_t) P_full };
        ggml_backend_tensor_set(lod_meta, meta, 0, sizeof(meta));
    }

    if (lod_meanidx) {
        std::vector<int32_t> idx(lod_meanidx->ne[0]);
        for (int64_t i = 0; i < lod_meanidx->ne[0]; ++i) {
            idx[i] = (int32_t) i;
        }
        ggml_backend_tensor_set(lod_meanidx, idx.data(), 0, ggml_nbytes(lod_meanidx));
    }

    if (lod_pmeta) {
        const float pf = (float) P_full;
        ggml_backend_tensor_set(lod_pmeta, &pf, 0, sizeof(pf));
    }

    if (lod_pageidx) {
        const uint32_t n_kv_h = lod_pageidx->ne[0]/n_full;
        std::vector<int32_t> idx(n_full*n_kv_h);
        for (uint32_t kv = 0; kv < n_kv_h; ++kv) {
            for (uint32_t f = 0; f < n_full; ++f) {
                idx[f + n_full*kv] = (int32_t)(prev_end/ps + f + P_cap*kv);
            }
        }
        ggml_backend_tensor_set(lod_pageidx, idx.data(), 0, ggml_nbytes(lod_pageidx));
    }

    if (lod_arange) {
        std::vector<int32_t> idx(n_exact_pad);
        for (uint32_t i = 0; i < n_exact_pad; ++i) {
            idx[i] = tail_start + i;
        }
        ggml_backend_tensor_set(lod_arange, idx.data(), 0, ggml_nbytes(lod_arange));
    }

    if (lod_catchup) {
        std::vector<int32_t> idx(catchup_len);
        for (uint32_t i = 0; i < catchup_len; ++i) {
            idx[i] = catchup_p0 + i;
        }
        ggml_backend_tensor_set(lod_catchup, idx.data(), 0, ggml_nbytes(lod_catchup));
    }

    // LLAMA_LOD_CHECK=1: verify the maintained page sums against the cache before this
    // ubatch folds into them - the first ubatch that reports CORRUPT names the edit that
    // desynchronised them (prompt-cache truncation, state restore, context shift, ...)
    static const char * check_env = getenv("LLAMA_LOD_CHECK");
    if (check_env != nullptr && atoi(check_env) != 0 && n_tokens > 1) {
        // LLAMA_LOD_CHECK=2 first corrupts one page on purpose: a detector that has never
        // fired is not evidence of anything
        if (atoi(check_env) > 1) {
            mctx->lod_corrupt_one_page();
        }
        mctx->lod_check_sums(std::min(prev_end, mctx->get_lod_sums_pos()), "prefill");
    }

    // LLAMA_LOD_DBG_UBATCH=1: one line per ubatch with the geometry the read was built
    // from. Diffing this between b==ub and b>ub shows whether a split batch drives the
    // geometry somewhere different, or whether the geometry is identical and the loss is
    // downstream of it.
    static const char * dbg_ub = getenv("LLAMA_LOD_DBG_UBATCH");
    if (dbg_ub != nullptr && atoi(dbg_ub) != 0) {
        LLAMA_LOG_WARN("lod_ub: pos0=%d n_tok=%u prev_end=%u P_full=%u tail=%u kP=%u "
                "n_exact=%u catchup=%u sums_pos=%u stat_fa=%d P_cap=%u\n",
                (int) ubatch->pos[0], n_tokens, prev_end, P_full, tail_start, kP,
                n_exact_pad, catchup_len, mctx->get_lod_sums_pos(), stat_fa ? 1 : 0, P_cap);
    }

    // after this graph runs, the sums cover everything up to the end of the ubatch
    mctx->lod_note_sums(prev_end + n_tokens);
}

bool llm_graph_input_attn_lod::can_reuse(const llm_graph_params & params) {
    const llama_kv_cache_context * base = nullptr;
    switch (parent) {
        case LOD_PARENT_ISWA:   base = static_cast<const llama_kv_cache_iswa_context *>(params.mctx)->get_base(); break;
        case LOD_PARENT_HYBRID: base = static_cast<const llama_memory_hybrid_context *>(params.mctx)->get_attn(); break;
        case LOD_PARENT_PLAIN:  base = static_cast<const llama_kv_cache_context      *>(params.mctx);             break;
    }

    this->mctx = base;

    const uint32_t prev_end_new = base->get_head();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= !lod_mask || lod_mask->ne[1] == (int64_t) params.ubatch.n_tokens;

    const uint32_t P_new = prev_end_new / ps;

    res &= base->is_single_stream_contig();
    res &= base->get_stream() == strm; // the views bake this stream's byte offsets

    // reuse and pipeline parallelism are mutually exclusive: a reused graph has a single
    // input-buffer set, so llama_context must synchronize the scheduler before filling it,
    // and that stalls every device once per ubatch. Prefer the rebuild path, exactly like
    // dense prefill does. (This guard used to sit inside the static-graph branch only, so
    // any ubatch that missed the page-aligned static path - i.e. most real prompts - lost
    // multi-GPU prefill scaling entirely.)
    res &= !cparams.pipeline_parallel;
    res &= base->get_lod_sums_pos() >= prev_end_new; // no pending catch-up
    res &= catchup_len == 0;
    const uint32_t top_max = dec ? cparams.lod_top_pages_dec : cparams.lod_top_pages;
    res &= std::min(top_max, P_new) == kP;

    // the fused path takes its geometry from lod_meta at run time; the other paths bake
    // view offsets into the graph and only hold within one page window
    if (!lod_meta && !stat_fa) {
        res &= P_new == P_full;
    }

    if (stat_fa && mask_S > 0) {
        // the mask-direct graph bakes the fattn span; rebuild at span boundaries
        static const char * span_env = getenv("LLAMA_LOD_SPAN");
        const uint32_t span_step = span_env != nullptr ? (uint32_t) std::max(0, atoi(span_env)) : 0;
        const uint32_t S_new = span_step > 0
            ? std::min<uint32_t>(GGML_PAD(prev_end_new + params.ubatch.n_tokens, span_step), cparams.n_ctx_seq)
            : cparams.n_ctx_seq;
        res &= S_new == mask_S;
    }

    // the padded page capacity is baked into every static-graph shape
    res &= lod_page_capacity(prev_end_new, params.ubatch.n_tokens, ps, cparams.n_ctx_seq) == P_cap;

    if (stat_fa) {
        // capacity-padded graph: any aligned ubatch with a saturated budget reuses.
        // under pipeline parallelism reuse forces a scheduler synchronize per ubatch
        // (single input-buffer set), which defeats the multi-GPU pipeline - prefer
        // the rebuild path there, exactly like dense prefill
        res &= prev_end_new % ps == 0;
        res &= params.ubatch.n_tokens % ps == 0;
        res &= P_new >= top_max;
        res &= P_new*ps + n_exact_pad <= cparams.n_ctx_seq;
    }

    if (res) {
        prev_end   = prev_end_new;
        P_full     = P_new;
        tail_start = P_new*ps;
    }

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    // with LoD every attention layer reads through its own input, leaving these
    // unconsumed and therefore unallocated
    if (inp_attn->self_k_idxs->buffer) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot && inp_attn->self_k_rot->buffer) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot && inp_attn->self_v_rot->buffer) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, tmp);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_moe_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, up);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    if (hparams.f_embedding_scale != 0.0f && (ubatch.token || hparams.n_deepstack_layers == 0)) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    GGML_UNUSED(v_cur);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = k;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

void llm_graph_input_attn_lod2::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    std::vector<int32_t> meta(4*groups.size(), 0);

    for (size_t i = 0; i < groups.size(); ++i) {
        const auto & g = groups[i];

        // the archive is indexed by cell, and LoD2 reads it by position, so the
        // two must coincide: append only, no shift, within the group's stream
        // (spec 8.5)
        GGML_ASSERT((uint32_t) ubatch->pos[g.t0] == g.p0);

        // and the group has to be exactly one sequence's run of queries, since
        // its ops read that stream's archive and state
        for (uint32_t t = 0; t < g.nt; ++t) {
            const uint32_t j = g.t0 + t;
            if (ubatch->n_seq_id[j] != 1 ||
                ubatch->seq_id[j][0] != ubatch->seq_id[g.t0][0] ||
                (uint32_t) ubatch->pos[j] != g.p0 + t) {
                LLAMA_LOG_ERROR("lod2: group %zu strm=%u t0=%u nt=%u p0=%u : token %u "
                        "n_seq_id=%d seq=%d pos=%d\n", i, g.strm, g.t0, g.nt, g.p0, j,
                        ubatch->n_seq_id[j], ubatch->seq_id[j][0], ubatch->pos[j]);
                GGML_ABORT("LoD2: ubatch group is not a single contiguous sequence");
            }
        }

        meta[4*i] = (int32_t) g.p0;
    }

    if (lod2_meta && !groups.empty()) {
        ggml_backend_tensor_set(lod2_meta, meta.data(), 0, meta.size()*sizeof(int32_t));
    }

    // The plan was decided when the graph was built; publishing it here means a
    // graph that is reserved but never computed leaves the cache state alone.
    for (const auto & g : groups) {
        mctx->set_lod2_state(g.strm, g.coverage_post, g.slots_post);
    }
}

bool llm_graph_input_attn_lod2::can_reuse(const llm_graph_params & params) {
    // Off by default: two earlier attempts produced wrong output (see
    // docs/lod2-port-handover.md).  LLAMA_LOD2_REUSE=1 turns it on and
    // LLAMA_LOD2_REUSE=2 additionally traces every decision, because the
    // failure mode reported so far - correct at -ub 512, no output at -ub 4096
    // - is not something any check here looks at.
    static const int reuse = getenv("LLAMA_LOD2_REUSE") ? atoi(getenv("LLAMA_LOD2_REUSE")) : 0;
    if (reuse == 0) {
        return false;
    }

    const auto * base = static_cast<const llama_memory_hybrid_context *>(params.mctx)->get_attn();

    const uint32_t p0_new = base->get_head();
    const uint32_t p1_new = p0_new + params.ubatch.n_tokens;
    const uint32_t cov    = base->get_lod2_coverage();
    const uint32_t slots  = base->get_lod2_slots();

    const auto & p = base->get_lod2_params();
    const uint32_t target = params.ubatch.n_tokens > 1
        ? p.local_begin(p0_new)
        : p.decode_begin(p1_new);

    const char * why = nullptr;
    if (groups.size() != 1)                          { why = "multi-group";    }
    else if (!base->is_single_stream_contig())       { why = "multi-stream";   }
    else if (params.ubatch.n_tokens != p1 - p0)      { why = "width";          }
    else if (cov != groups[0].coverage_post)         { why = "coverage";       }
    else if (slots != groups[0].slots_post)          { why = "slots";          }
    else if (target != cov)                          { why = "update due";     }

    if (reuse > 1) {
        LLAMA_LOG_INFO("lod2 reuse: n_tok=%u p0 %u->%u cov=%u/%u slots=%u/%u target=%u : %s\n",
                params.ubatch.n_tokens, p0, p0_new, cov,
                groups.empty() ? 0 : groups[0].coverage_post,
                slots, groups.empty() ? 0 : groups[0].slots_post,
                target, why ? why : "REUSE");
    }

    if (why) {
        return false;
    }

    p0   = p0_new;
    p1   = p1_new;
    groups[0].p0 = p0_new;
    groups[0].p1 = p1_new;
    mctx = base;

    return true;
}

llm_graph_input_attn_lod2 * llm_graph_context::build_attn_inp_lod2(const llama_kv_cache_context * mctx_base) const {
    GGML_ASSERT(mctx_base->is_lod2());

    // Every group has to be an append-only run of cells, because LoD2 indexes
    // the archive by position.  Groups themselves are fine: with one KV stream
    // per sequence they are separate archives that share nothing.
    if (!mctx_base->lod2_groups_contiguous()) {
        return nullptr;
    }

    auto inp = std::make_unique<llm_graph_input_attn_lod2>(cparams, mctx_base);

    const uint32_t ng = mctx_base->get_lod2_n_groups();

    // Group widths come from the cell lists when they describe this ubatch.
    // The context that graph_reserve builds carries a dummy slot info of one
    // cell per stream whatever the reserve ubatch is; that graph is only ever
    // measured, never computed, so there the tokens are divided evenly and the
    // state is taken as empty.  Reading the live coverage against the dummy's
    // p0 of zero is what it would otherwise do, and an empty state is also the
    // widest plan, which is what a reservation wants.
    uint32_t n_cells = 0;
    for (uint32_t i = 0; i < ng; ++i) {
        n_cells += mctx_base->get_lod2_group_len(i);
    }
    const bool real = ng > 0 && n_cells == n_tokens;

    uint32_t t0 = 0;
    for (uint32_t i = 0; i < ng; ++i) {
        llm_graph_input_attn_lod2::group g;

        g.strm = mctx_base->get_lod2_group_stream(i);
        g.t0   = t0;
        g.nt   = real ? mctx_base->get_lod2_group_len(i)
                      : (i + 1 == ng ? n_tokens - t0 : n_tokens/ng);
        g.p0   = real ? mctx_base->get_lod2_group_head(i) : 0;
        g.p1   = g.p0 + g.nt;

        g.coverage_pre  = real ? mctx_base->get_lod2_coverage(g.strm) : 0;
        g.slots_pre     = real ? mctx_base->get_lod2_slots   (g.strm) : 0;
        g.coverage_post = g.coverage_pre;
        g.slots_post    = g.slots_pre;

        t0 += g.nt;

        inp->groups.push_back(g);
    }
    GGML_ASSERT(t0 == n_tokens);

    inp->p0 = inp->groups.empty() ? 0 : inp->groups[0].p0;
    inp->p1 = inp->p0 + (inp->groups.empty() ? 0 : inp->groups[0].nt);

    inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
    inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

    // one meta tensor for every layer: the schedule is per sequence, not per
    // layer, so it holds one entry per group
    inp->lod2_meta = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 4, std::max<int64_t>(1, ng));
    ggml_set_input(inp->lod2_meta);
    ggml_set_name(inp->lod2_meta, "lod2_meta");

    return (llm_graph_input_attn_lod2 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_lod2 * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(kq_b  == nullptr && "LoD2 does not support KQ bias");
    GGML_ASSERT(sinks == nullptr && "LoD2 does not support sinks");
    GGML_ASSERT(v_mla == nullptr);

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto * mctx_cur = inp->mctx;

    // the archive is the cache: write this ubatch before anything reads it
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    const auto & p = mctx_cur->get_lod2_params();

    // q as the op wants it: [D, n_tokens, n_head_q]
    ggml_tensor * q_all = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);

    const size_t ng = inp->groups.size();
    GGML_ASSERT(ng > 0);

    ggml_tensor * cur = nullptr;

    // One chain per group.  The groups write disjoint state planes and read
    // disjoint archives, so nothing orders them against each other; only the
    // updates within a group are chained.
    for (size_t gi = 0; gi < ng; ++gi) {
        auto & G = inp->groups[gi];

        const uint32_t p0 = G.p0;
        const uint32_t p1 = G.p1;

        ggml_tensor * k     = mctx_cur->get_lod2_k    (ctx0, il, G.strm);
        ggml_tensor * v     = mctx_cur->get_lod2_v    (ctx0, il, G.strm);
        ggml_tensor * s_kv  = mctx_cur->get_lod2_s_kv (ctx0, il, G.strm);
        ggml_tensor * s_mn  = mctx_cur->get_lod2_s_mn (ctx0, il, G.strm);
        ggml_tensor * p_kv  = mctx_cur->get_lod2_p_kv (ctx0, il, G.strm);
        ggml_tensor * p_idx = mctx_cur->get_lod2_p_idx(ctx0, il, G.strm);
        ggml_tensor * s_pg  = mctx_cur->get_lod2_s_pg (ctx0, il, G.strm);
        ggml_tensor * meta  = mctx_cur->get_lod2_meta (ctx0, il, G.strm);

        ggml_tensor * q = q_all;
        if (ng > 1) {
            q = ggml_cont(ctx0, ggml_view_3d(ctx0, q_all,
                    q_all->ne[0], G.nt, q_all->ne[2],
                    q_all->nb[1], q_all->nb[2], (size_t) G.t0*q_all->nb[1]));
        }

        uint32_t coverage = G.coverage_pre;
        uint32_t slots    = G.slots_pre;

        // Chaining each update's result into the next op's state src is what
        // orders the nodes: they all write the same buffers, so without the
        // dependency the scheduler would be free to interleave them.
        ggml_tensor * state = s_kv;

        const auto emit_updates = [&](const std::vector<llama_lod2_block> & blocks) {
            for (const auto & b : blocks) {
                state = ggml_lod2_update(ctx0, k, v, state, s_mn, p_kv, p_idx, s_pg, meta,
                        b.p0, b.p1, b.s_len, b.s_len_new, p.sink_len);
                ggml_build_forward_expand(gf, state);

                coverage = b.p1;
                slots    = b.s_len_new;
            }
        };

        // Absorb everything this group will not read exactly.  Prefill measures
        // the window from the group start with the prefill lookback; generation
        // uses the smaller decode window.  The two schedules differ in step size
        // and in the context the growth budget is measured against
        // (spec 4, 6.1, 6.2).
        if (G.nt > 1) {
            // Padded to a fixed length so that the node count of this graph does
            // not depend on where the schedule happens to land.  A varying node
            // count makes ggml_gallocr reallocate every ubatch, and the
            // reallocation path synchronizes every backend - which drains the
            // scheduler's pipeline and is why this path used to get no multi-GPU
            // prefill scaling at all.  The padding blocks are empty (p0 == p1)
            // and every backend skips them.  The bound is taken from the ubatch
            // size, not from this group's token count: a group advances coverage
            // by however many tokens the PREVIOUS one carried, so a short tail
            // still has to schedule a full one's worth of blocks.  A plan longer
            // than the bound is left unpadded rather than refused - it costs one
            // reallocation, not correctness.
            auto blocks = llama_lod2_plan_prefill(p, coverage, slots, p.local_begin(p0));
            const size_t want = p.prefill_blocks_max(p.prefill_chunk);
            while (blocks.size() < want) {
                const uint32_t c = blocks.empty() ? coverage : blocks.back().p1;
                const uint32_t sl = blocks.empty() ? slots   : blocks.back().s_len_new;
                blocks.push_back({ c, c, sl, sl });
            }
            emit_updates(blocks);
        } else {
            emit_updates(llama_lod2_plan_tail(p, coverage, slots,
                        p.decode_begin(p1), p1, p.decode_state_update));
        }

        // The exact window starts exactly where the state ends: the two branches
        // partition the history, so any other boundary would drop or double count
        // tokens.  This is the invariant, not the planned target.
        const uint32_t l0 = coverage;
        if (l0 > p0) {
            LLAMA_LOG_ERROR("lod2: state ahead of queries: il=%d group %zu/%zu strm=%u "
                    "t0=%u nt=%u p0=%u p1=%u cov_pre=%u slots_pre=%u l0=%u slots=%u n_tokens=%u\n",
                    il, gi, ng, G.strm, G.t0, G.nt, p0, p1,
                    G.coverage_pre, G.slots_pre, l0, slots, n_tokens);
        }
        GGML_ASSERT(l0 <= p0 && "LoD2 state ran ahead of the queries");

        const uint32_t routes = G.nt > 1 ? p.routes_prefill : p.routes_decode;

        // The routing logits are q . mean_k for every live slot.  That
        // contraction is the dominant cost of prefill and it is a GEMM: handing
        // it to ggml_mul_mat puts it on the matrix units instead of a per-query
        // loop, which is exactly what the author's Triton path does (it passes
        // finished route logits into its fused kernel).  The mean table is
        // copied to a compact [D, slots, H_kv] first: one small copy per layer,
        // against a strided src0 that the backends would rather not see.
        //
        // Generation is the opposite case.  With one query the GEMM degenerates
        // to a GEMV that reads the mean table once per query head, and the copy
        // that feeds it costs a read plus a write of the whole table - together
        // 0.98 ms per token against the 0.09 ms the traffic is worth.  The op
        // reads the table once per KV head instead, so the whole pair comes out
        // of the graph.
        //
        // The view is taken at the full slot capacity rather than at the live
        // slot count, even though only the first 'slots' columns are ever read.
        // A view that grows with the state makes this GEMM's operands larger
        // every ubatch, which takes them past what graph_reserve sized (slots is
        // 0 there) and forces a graph reallocation - the same pipeline drain the
        // block padding above avoids.  Every consumer already takes the slot
        // count as an op parameter and the row stride as lgt->ne[0], so nothing
        // else changes.
        ggml_tensor * logits = nullptr;
        if (G.nt > 1) {
            ggml_tensor * mk = ggml_view_3d(ctx0, s_mn,
                    k->ne[0], s_mn->ne[1], s_mn->ne[2], s_mn->nb[1], s_mn->nb[2], 0);
            logits = ggml_mul_mat(ctx0, ggml_cont(ctx0, mk), q); // [S_cap, nt, n_head_q]
        }

        ggml_tensor * gmeta = inp->lod2_meta;
        if (ng > 1) {
            gmeta = ggml_view_1d(ctx0, inp->lod2_meta, 4, (size_t) gi*inp->lod2_meta->nb[1]);
        }

        ggml_tensor * out = ggml_lod2_attn(ctx0, q, k, v, state, s_mn, p_kv, p_idx, s_pg,
                gmeta, logits, l0, slots, routes, p.sink_len, kq_scale);
        ggml_build_forward_expand(gf, out);

        G.coverage_post = coverage;
        G.slots_post    = slots;

        // the op returns [Dv, n_head_q, nt], so the groups stack along dim 2
        cur = cur ? ggml_concat(ctx0, cur, out, 2) : out;
    }

    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], n_tokens);
    cb(cur, "kqv_out_lod2", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_lod * llm_graph_context::build_attn_inp_lod(const llama_kv_cache_context * mctx_base, llm_graph_input_attn_lod::lod_parent_t parent) const {
    GGML_ASSERT(mctx_base->is_lod());
    GGML_ASSERT(cparams.lod_top_pages > 0);

    // mixed-stream ubatches (several server slots batched together) fall back to the
    // dense path - coherent, since the cache holds exactly what dense would hold
    if (!mctx_base->is_single_stream_contig()) {
        return nullptr;
    }

    auto inp = std::make_unique<llm_graph_input_attn_lod>(hparams, cparams, mctx_base);

    const uint32_t ps       = mctx_base->get_lod_page_size();
    const uint32_t prev_end = mctx_base->get_head();

    inp->parent     = parent;
    inp->ps         = ps;
    inp->strm       = mctx_base->get_stream();
    inp->prev_end   = prev_end;
    inp->P_full     = prev_end / ps;
    inp->tail_start = inp->P_full * ps;
    // one token per query at decode, a whole ubatch sharing one page set at prefill: the
    // phases get separate budgets (see llama_cparams)
    inp->dec        = n_tokens <= 8;
    const uint32_t top_max_inp = inp->dec ? cparams.lod_top_pages_dec : cparams.lod_top_pages;
    inp->kP         = std::min(top_max_inp, inp->P_full);
    inp->NL         = inp->kP * ps;

    // reserve/probe graphs run set_input on dummy state, so the counter can be ahead;
    // ahead is harmless (the current-ubatch fold always runs from prev_end)
    inp->catchup_p0  = std::min(mctx_base->get_lod_sums_pos(), prev_end);
    inp->catchup_len = prev_end - inp->catchup_p0;

    // padded so that the graph shape stays fixed while decode fills the current page
    inp->n_exact_pad = std::min<uint32_t>(ps + n_tokens, cparams.n_ctx_seq - inp->tail_start);
    GGML_ASSERT(inp->n_exact_pad >= prev_end + n_tokens - inp->tail_start);

    inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
    inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

    // only allocated when consumed: the fused read needs no mask, unused inputs have no buffer
    // q8_0 caches ride the same paths as F16: fattn reads them via its internal
    // dequant, the fused decode kernel dequantizes q8_0 blocks in-kernel
    const auto lod_type_ok = [](ggml_type t) {
        return !ggml_is_quantized(t) || t == GGML_TYPE_Q8_0;
    };
    const bool use_fa_pre = cparams.flash_attn && n_tokens > 8 &&
            (mctx_base->type_k() == GGML_TYPE_F16 || mctx_base->type_k() == GGML_TYPE_Q8_0) &&
            (mctx_base->type_v() == GGML_TYPE_F16 || mctx_base->type_v() == GGML_TYPE_Q8_0);
    const bool use_fused_pre = !use_fa_pre && cparams.lod_fused && inp->kP > 0 && n_tokens <= 8 &&
            lod_type_ok(mctx_base->type_k()) && lod_type_ok(mctx_base->type_v());

    // the mask is generated on-device in build_attn (regular structure, no upload)

    // catchup_len == 1 folds through the single-token shortcut which does not read
    // lod_ones - creating it would leave an unconsumed input without a buffer
    if ((n_tokens > 1 && !use_fused_pre) || inp->catchup_len > 1) {
        // the fused op folds the sums itself; single-token updates reduce to a plain add
        inp->lod_ones = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ps);
        ggml_set_input(inp->lod_ones);
        ggml_set_name(inp->lod_ones, "lod_ones");
    }

    if (inp->catchup_len > 0) {
        inp->lod_catchup = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->catchup_len);
        ggml_set_input(inp->lod_catchup);
        ggml_set_name(inp->lod_catchup, "lod_catchup");
    }

    if (use_fused_pre) {
        inp->lod_meta = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2); // {n_past, n_pages}
        ggml_set_input(inp->lod_meta);
        ggml_set_name(inp->lod_meta, "lod_meta");
    }

    // static prefill graph: capacity-padded shapes so the graph is reusable across
    // ubatches and the scheduler pipelines like dense. Gated on a saturated page
    // budget (top-k can then never pick an invalid page) and page alignment
    inp->P_cap  = llm_graph_input_attn_lod::lod_page_capacity(prev_end, n_tokens, ps, cparams.n_ctx_seq);
    inp->n_full = n_tokens / ps;
    inp->stat_fa = use_fa_pre && inp->kP == top_max_inp &&
            prev_end % ps == 0 && n_tokens % ps == 0 && inp->catchup_len == 0 &&
            inp->n_exact_pad == ps + n_tokens;

    // mask-direct prefill (ablation env LLAMA_LOD_PREFILL=mask): page means are
    // refreshed from the sums every prefill ubatch and read as extra fattn columns
    // mask-direct prefill needs no gather at all - see use_mask_pre below
    static const char * prefill_env_pre = getenv("LLAMA_LOD_PREFILL");
    const bool mask_pre = use_fa_pre && prefill_env_pre != nullptr && strcmp(prefill_env_pre, "mask") == 0 &&
            mctx_base->has_lod_means() && inp->kP > 0;

    if (mask_pre) {
        inp->lod_meanidx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->P_cap);
        ggml_set_input(inp->lod_meanidx);
        ggml_set_name(inp->lod_meanidx, "lod_meanidx");
    }

    if (inp->stat_fa) {
        inp->lod_pmeta = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_input(inp->lod_pmeta);
        ggml_set_name(inp->lod_pmeta, "lod_pmeta");

        // KV head count of the LoD (non-SWA) layers - layer 0 may be SWA with a
        // different head count
        uint32_t hkv_lod = hparams.n_head_kv(0);
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!hparams.is_swa(il)) {
                hkv_lod = hparams.n_head_kv(il);
                break;
            }
        }

        inp->lod_pageidx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->n_full*hkv_lod);
        ggml_set_input(inp->lod_pageidx);
        ggml_set_name(inp->lod_pageidx, "lod_pageidx");
    }

    // the exact tier reads through a gather: dequantizes quantized caches, and keeps
    // the FA graph shape static across ubatches (the moving tail window lives in the
    // index input, not in view offsets)
    if ((((ggml_is_quantized(mctx_base->type_k()) || ggml_is_quantized(mctx_base->type_v())) && !use_fused_pre) || use_fa_pre) && !mask_pre) {
        inp->lod_arange = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->n_exact_pad);
        ggml_set_input(inp->lod_arange);
        ggml_set_name(inp->lod_arange, "lod_arange");
    }

    return (llm_graph_input_attn_lod *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_lod * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(kq_b  == nullptr && "LoD attention does not support KQ bias");
    GGML_ASSERT(sinks == nullptr && "LoD attention does not support sinks yet");
    GGML_ASSERT(v_mla == nullptr);

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    const uint32_t ps         = inp->ps;
    const uint32_t prev_end   = inp->prev_end;
    const uint32_t P_full     = inp->P_full;
    const uint32_t tail_start = inp->tail_start;
    const uint32_t n_exact    = inp->n_exact_pad;
    // per-layer budget (inp->kP is the max-based value used by the shared gates)
    const auto & top_pages_l   = inp->dec ? cparams.lod_top_pages_dec_l   : cparams.lod_top_pages_l;
    const auto & top_regions_l = inp->dec ? cparams.lod_top_regions_dec_l : cparams.lod_top_regions_l;

    const uint32_t kP         = std::min<uint32_t>(top_pages_l[il], P_full);
    const uint32_t NL         = kP*ps;

    const int64_t n_head_kv      = k_cur->ne[1];
    const int64_t n_embd_head_k  = k_cur->ne[0];
    const int64_t n_embd_head_v  = v_cur->ne[0];
    const int64_t n_head_q       = q_cur->ne[1];

    // FA path: one flash-attention call over the assembled [leaf | exact | summary] KV set
    // (requires an F16 cache; silencing lives in the mask, which forces layer-shared selection)
    // decode stays on the tiered composition - the per-token KV assembly costs more than it saves
    const bool q_cache = ggml_is_quantized(mctx_cur->type_k()) || ggml_is_quantized(mctx_cur->type_v());

    const bool use_fa = cparams.flash_attn && n_tokens > 8 &&
            (q_cache || (mctx_cur->type_k() == GGML_TYPE_F16 && mctx_cur->type_v() == GGML_TYPE_F16));

    // fused single-kernel read (decode): reads the leaves straight from the cache, no gather
    const auto lod_type_ok = [](ggml_type t) {
        return !ggml_is_quantized(t) || t == GGML_TYPE_Q8_0;
    };
    const bool use_fused = !use_fa && cparams.lod_fused && kP > 0 && n_tokens <= 8 && n_embd_head_v <= 1024 &&
            lod_type_ok(mctx_cur->type_k()) && lod_type_ok(mctx_cur->type_v());

    // LLAMA_LOD_DBG_SHAPES=1: one line per layer naming the geometry the graph is being
    // built for. When a shape assert fires, the last line printed IS the failing case.
    if (getenv("LLAMA_LOD_DBG_SHAPES") != nullptr) {
        LLAMA_LOG_INFO("lod dbg: il=%d n_tokens=%d prev_end=%u P_full=%u P_cap=%u kP=%u "
                "n_exact=%u catchup=%u fa=%d fused=%d\n",
                il, (int) n_tokens, prev_end, P_full, inp->P_cap, kP,
                n_exact, inp->catchup_len, (int) use_fa, (int) use_fused);
    }

    // fold this ubatch into the page sums; sums accumulate from the raw (pre-quantization) K/V
    {
        auto update_sums = [&](ggml_tensor * cur, bool is_v, uint32_t t_start, uint32_t t_len) {
            auto get_sums = [&](uint32_t p0, uint32_t np) {
                return is_v ? mctx_cur->get_v_page_sums(ctx0, il, p0, np)
                            : mctx_cur->get_k_page_sums(ctx0, il, p0, np);
            };
            const int64_t D = cur->ne[0];

            // sum a token range of cur over the token axis -> [D, 1, n_head_kv]
            auto range_sum = [&](uint32_t t0, uint32_t len) {
                GGML_ASSERT(t0 + len <= t_len);
                ggml_tensor * slice = ggml_view_3d(ctx0, cur, D, n_head_kv, len, cur->nb[1], cur->nb[2], t0*cur->nb[2]);
                // mul_mat needs row-contiguous operands, so materialize the [len, D, n_head_kv] permutation
                ggml_tensor * perm  = ggml_cont(ctx0, ggml_permute(ctx0, slice, 1, 2, 0, 3));
                ggml_tensor * ones  = ggml_view_1d(ctx0, inp->lod_ones, len, 0);
                ggml_tensor * sums  = ggml_mul_mat(ctx0, ones, perm);        // [1, D, n_head_kv]
                return ggml_reshape_3d(ctx0, sums, D, 1, n_head_kv);
            };

            // partial edge page: read-modify-write its running sum
            auto edge = [&](uint32_t p, uint32_t t0g, uint32_t t1g) {
                ggml_tensor * part = range_sum(t0g - t_start, t1g - t0g);
                ggml_tensor * old  = ggml_cont(ctx0, get_sums(p, 1));
                ggml_tensor * next = ggml_add(ctx0, part, old);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, next, get_sums(p, 1)));
            };

            const uint32_t t_end = t_start + t_len;

            const uint32_t pa  = t_start / ps;        // page the range starts in
            const uint32_t pt  = t_end / ps;          // page the range ends in (exclusive end)
            const uint32_t pfa = (t_start + ps - 1) / ps;
            const uint32_t pfb = t_end / ps;

            if (t_len == 1) {
                // decode: one token lands in one page, the update is a plain add
                ggml_tensor * dst_row = get_sums(pa, 1);                              // [D, 1, Hkv]
                ggml_tensor * add_row = ggml_reshape_3d(ctx0, cur, D, 1, n_head_kv);  // same layout
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_add(ctx0, add_row, ggml_cont(ctx0, dst_row)), dst_row));
                return;
            }

            if (t_start % ps != 0 && pa == (t_end - 1) / ps) {
                // the whole range lands inside one already-started page
                edge(pa, t_start, t_end);
                return;
            }

            if (t_start % ps != 0) {
                edge(pa, t_start, pfa*ps);
            }

            if (pfb > pfa) {
                // pages fully covered by the range: one batched reduction
                const uint32_t nfull = pfb - pfa;
                const uint32_t l0    = pfa*ps - t_start;

                ggml_tensor * region = ggml_view_3d(ctx0, cur, D, n_head_kv, nfull*ps, cur->nb[1], cur->nb[2], l0*cur->nb[2]);
                region = ggml_reshape_4d(ctx0, region, D, n_head_kv, ps, nfull);
                region = ggml_cont(ctx0, ggml_permute(ctx0, region, 1, 2, 0, 3)); // [ps, D, n_head_kv, nfull]

                ggml_tensor * sums = ggml_mul_mat(ctx0, inp->lod_ones, region); // [1, D, n_head_kv, nfull]
                sums = ggml_reshape_3d(ctx0, sums, D, n_head_kv, nfull);
                sums = ggml_cont(ctx0, ggml_permute(ctx0, sums, 0, 2, 1, 3));   // [D, nfull, n_head_kv]

                if (inp->stat_fa) {
                    // static graph: the destination pages travel in an index input
                    ggml_tensor * rows = mctx_cur->get_k_page_sumrows(ctx0, il);
                    if (is_v) {
                        rows = mctx_cur->get_v_page_sumrows(ctx0, il);
                    }
                    ggml_tensor * src2 = ggml_reshape_2d(ctx0, sums, D, (int64_t) nfull*n_head_kv);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, rows, src2, inp->lod_pageidx));
                } else {
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, sums, get_sums(pfa, nfull)));
                }
            }

            if (t_end % ps != 0 && pt >= pfa) {
                // freshly started trailing page (its old sum is zero, RMW keeps it general)
                edge(pt, pt*ps, t_end);
            }
        };

        // lazy catch-up: fold tokens processed by dense-path ubatches (other server slots
        // batched in) from the cache before folding the current ubatch
        if (inp->catchup_len > 0) {
            ggml_tensor * ck = ggml_get_rows(ctx0, mctx_cur->get_k_tokrows(ctx0, il, cparams.n_ctx_seq), inp->lod_catchup);
            ggml_tensor * cv = ggml_get_rows(ctx0, mctx_cur->get_v_tokrows(ctx0, il, cparams.n_ctx_seq), inp->lod_catchup);

            const int64_t Dk = k_cur->ne[0];
            const int64_t Dv = v_cur->ne[0];

            update_sums(ggml_reshape_3d(ctx0, ck, Dk, n_head_kv, inp->catchup_len), false, inp->catchup_p0, inp->catchup_len);
            update_sums(ggml_reshape_3d(ctx0, cv, Dv, n_head_kv, inp->catchup_len), true,  inp->catchup_p0, inp->catchup_len);
        }

        if (!use_fused) {
            update_sums(k_cur, false, prev_end, n_tokens);
            update_sums(v_cur, true,  prev_end, n_tokens);
        }
    }

    // mask-direct prefill: refresh ALL page-mean rows from the sums (correct after any
    // decode/rollback history - the sums are authoritative; ~0.03% of an ubatch)
    if (inp->lod_meanidx != nullptr && use_fa) {
        const uint32_t P_cap = inp->P_cap;

        auto refresh = [&](ggml_tensor * sums, ggml_tensor * rows) {
            ggml_tensor * t = ggml_cont(ctx0, ggml_permute(ctx0, sums, 0, 2, 1, 3)); // [D, Hkv, P_cap]
            t = ggml_reshape_2d(ctx0, t, t->ne[0]*t->ne[1], P_cap);                  // cache row layout
            t = ggml_scale(ctx0, t, 1.0f/ps);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, rows, t, inp->lod_meanidx));
        };
        refresh(mctx_cur->get_k_page_sums(ctx0, il, 0, P_cap), mctx_cur->get_k_meanrows(ctx0, il, P_cap));
        refresh(mctx_cur->get_v_page_sums(ctx0, il, 0, P_cap), mctx_cur->get_v_meanrows(ctx0, il, P_cap));
    }

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3); // [D_k, n_tokens, n_head_q]

    // LLAMA_LOD_PROFILE=1: emit, per layer, the true dense attention mass of each complete
    // page and the LoD page score for the same query. Ranking the pages by score and taking
    // the cumulative mass gives capture(k) for EVERY budget k from this one pass, which is
    // what a per-layer budget search needs (run it with a saturating budget so the read is
    // exactly dense). Emitted for the last query of the ubatch - the most recent context,
    // and the only query at decode.
    // read per build, not once: llama-lod-search turns profiling off after it has used one
    // profiled pass to discover the LoD layers, and a cached flag would keep the branch (and
    // its cost) in every later graph
    const char * prof_env = getenv("LLAMA_LOD_PROFILE");
    if (prof_env != nullptr && atoi(prof_env) != 0 && P_full > 0 && n_tokens > 1) {
        const uint32_t T = P_full*ps; // complete pages only: visible to EVERY query here

        // Sample queries across the ubatch rather than profiling one. Prefill picks ONE
        // page set for the whole ubatch, so what decides whether a budget is enough is the
        // union of what all those queries want - profiling the last query alone measures a
        // distribution dominated by recent context, which the exact tail already covers,
        // and produces budgets that lose badly on real retrieval (measured 3/12 vs 11/12).
        // LLAMA_LOD_PROFILE=1 samples queries spread across the ubatch, which is what a
        // gather-mode selection has to serve with ONE page set. =2 samples 32 CONSECUTIVE
        // queries instead, i.e. exactly one mask-mode selection block, so the two runs
        // measure what the per-block granularity is worth at the same budget.
        const bool    block_mode = atoi(prof_env) >= 2;
        const int64_t n_samp = std::min<int64_t>(32, n_tokens);
        const int64_t stride = block_mode ? 1 : std::max<int64_t>(1, n_tokens/n_samp);
        const int64_t off    = block_mode ? std::max<int64_t>(0, n_tokens - n_samp) : 0;

        ggml_tensor * qs = ggml_cont(ctx0, ggml_view_3d(ctx0, q, n_embd_head_k, n_samp, n_head_q,
                stride*q->nb[1], q->nb[2], off*q->nb[1]));            // [D, n_samp, n_head_q]

        ggml_tensor * kall = mctx_cur->get_k_range(ctx0, il, 0, T);   // [D, T, n_head_kv]
        ggml_tensor * lg   = ggml_mul_mat(ctx0, kall, qs);            // [T, n_samp, n_head_q]
        ggml_mul_mat_set_prec(lg, GGML_PREC_F32);
        lg = ggml_soft_max_ext(ctx0, lg, nullptr, kq_scale, 0.0f);

        // sum the weights inside each page: avg-pool then undo the 1/ps
        ggml_tensor * pm = ggml_pool_2d(ctx0,
                ggml_reshape_3d(ctx0, lg, ps, P_full, n_samp*n_head_q),
                GGML_OP_POOL_AVG, ps, 1, ps, 1, 0, 0);                // [1, P_full, n_samp*n_head_q]
        pm = ggml_scale(ctx0, ggml_cont(ctx0, pm), (float) ps);
        pm = ggml_reshape_2d(ctx0, pm, P_full, n_samp*n_head_q);
        cb(pm, "lod_prof_mass", il);
        ggml_build_forward_expand(gf, pm);

        // the selector max-pools its score over queries and heads; emit the same axes so the
        // consumer reproduces exactly that pooling
        ggml_tensor * ps_score = ggml_mul_mat(ctx0,
                mctx_cur->get_k_page_sums(ctx0, il, 0, P_full), qs);  // [P_full, n_samp, n_head_q]
        ggml_mul_mat_set_prec(ps_score, GGML_PREC_F32);
        ps_score = ggml_reshape_2d(ctx0, ps_score, P_full, n_samp*n_head_q);
        cb(ps_score, "lod_prof_score", il);
        ggml_build_forward_expand(gf, ps_score);

        // =3 additionally emits the head-pooled score of EVERY query, so the consumer can
        // form each 32-query block's selection and measure how big their UNION is. That
        // number decides whether per-block selection can be served by a compact gather
        // (union x ps leaf rows) instead of reading the whole span the way mask does.
        if (atoi(prof_env) >= 3) {
            ggml_tensor * sa = ggml_mul_mat(ctx0,
                    mctx_cur->get_k_page_sums(ctx0, il, 0, P_full), q); // [P_full, n_tokens, n_head_q]
            ggml_mul_mat_set_prec(sa, GGML_PREC_F32);
            sa = ggml_cont(ctx0, ggml_permute(ctx0, sa, 1, 0, 2, 3));   // [n_tokens, P_full, n_head_q]
            sa = ggml_cont(ctx0, ggml_permute(ctx0, sa, 0, 2, 1, 3));   // [n_tokens, n_head_q, P_full]
            sa = ggml_pool_2d(ctx0, sa, GGML_OP_POOL_MAX, 1, n_head_q, 1, n_head_q, 0, 0);
            sa = ggml_reshape_2d(ctx0, ggml_cont(ctx0, sa), n_tokens, P_full);
            cb(sa, "lod_prof_qscore", il);
            ggml_build_forward_expand(gf, sa);
        }
    }

    // on-device mask parts (no CPU fill, no upload): the structure is fully regular.
    // exact tier: (i, j) visible iff tail_start + i <= prev_end + j; built from two
    // aranges and log(step(x)) for an exact 0 / -inf band
    // NL-sized parts are shareable across layers only under a uniform budget
    const bool kp_uniform = top_pages_l.empty() ||
            std::all_of(top_pages_l.begin(), top_pages_l.end(),
                    [&](uint32_t v) { return v == top_pages_l[0]; });

    ggml_tensor * m_band = inp->sh_band; // [n_exact_pad, n_tokens]
    ggml_tensor * m_leaf = kp_uniform ? inp->sh_leaf : nullptr; // [NL, n_tokens], zeros
    if (!use_fused && m_band == nullptr) {
        const float C = (float) prev_end - (float) tail_start;

        ggml_tensor * t_i = ggml_arange(ctx0, -C - 0.5f, -C - 0.5f + (float) n_exact, 1.0f); // i - C - 0.5
        ggml_tensor * a_j = ggml_arange(ctx0, 0.0f, (float) n_tokens, 1.0f);

        m_band = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, a_j, 1, n_tokens),
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_exact, n_tokens));
        m_band = ggml_sub(ctx0, m_band, ggml_reshape_2d(ctx0, t_i, n_exact, 1)); // j - i + C + 0.5
        m_band = ggml_log(ctx0, ggml_step(ctx0, m_band));

        inp->sh_band = m_band;

        if (kP > 0) {
            m_leaf = ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, NL, n_tokens), 0.0f);
            if (kp_uniform) {
                inp->sh_leaf = m_leaf;
            }
        }
    }
    if (!use_fused && m_leaf == nullptr && kP > 0) {
        m_leaf = ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, NL, n_tokens), 0.0f);
    }

    ggml_tensor * scores = nullptr;
    ggml_tensor * sel    = nullptr; // I32 [kP, n_head_kv] or [kP]
    ggml_tensor * lk     = nullptr; // [D_k, NL, n_head_kv]
    ggml_tensor * lv     = nullptr;

    // the fused op selects in-op (and over the runtime page count, so selection never
    // lags the static graph) - no selection subgraph at all
    // static prefill graphs read the page tier at full capacity; pages past the
    // runtime count are silenced with an exact -inf (log(step())) validity vector,
    // so the refinement denominator is untouched and top-k can never pick them
    // (the budget is saturated: kP real candidates always exist)
    const bool     stat_fa = inp->stat_fa;

    // Pad the page tier to the stepped capacity whenever the budget is saturated, not just
    // in the fully static path. Sizing it by the live page count instead makes the score
    // tensor grow by one page every ubatch, and a shape change forces the scheduler to
    // reallocate - which synchronises every device and shows up as pipeline bubbles, so
    // multi-GPU prefill ends up slower than a single GPU. Padding costs nothing at run time
    // (the extra pages are silenced exactly) and keeps the shape stable for ~256 pages.
    // Saturation is required: top-k must never be able to pick an invalid page, because the
    // leaf tier has no mask to silence one.
    const bool     pad_pages = kP > 0 && P_full >= kP;
    const uint32_t Pv        = (stat_fa || pad_pages) ? inp->P_cap : P_full;

    ggml_tensor * logv = inp->sh_logv; // [Pv]: 0 for valid pages, -inf past the runtime count
    if (Pv > P_full && logv == nullptr) {
        ggml_tensor * ar = ggml_arange(ctx0, 0.5f, (float) Pv + 0.5f, 1.0f);
        if (inp->lod_pmeta != nullptr) {
            // static graphs carry the runtime page count in an input
            ggml_tensor * pr = ggml_repeat(ctx0, inp->lod_pmeta, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, Pv));
            logv = ggml_log(ctx0, ggml_step(ctx0, ggml_sub(ctx0, pr, ar)));
        } else {
            // rebuilt graphs can bake it: only the value changes per ubatch, not the shape
            logv = ggml_log(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, ar, -1.0f, (float) P_full)));
        }
        inp->sh_logv = logv;
    }

    const bool use_mask_read = use_fa && inp->lod_meanidx != nullptr;

    if (kP > 0 && !use_fused && !use_mask_read) {
        // page scores at query-head resolution; 1/ps turns the stored sums into means
        ggml_tensor * kp = mctx_cur->get_k_page_sums(ctx0, il, 0, Pv);

        scores = ggml_mul_mat(ctx0, kp, q); // [Pv, n_tokens, n_head_q]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        if (!use_fused) {
            // sums vs means: ranking is scale-invariant, only the softmax path needs means
            scores = ggml_scale(ctx0, scores, 1.0f/ps);
        }
        cb(scores, "lod_scores", il);

        // selection shared by the query block: max over queries, then over heads
        // (per KV head when lod_sel_head, else one page set for the whole layer)
        const bool    sel_head = cparams.lod_sel_head && n_head_kv > 1 && !use_fa && !use_fused;
        const int64_t g        = n_head_q / n_head_kv;

        ggml_tensor * s1;
        if (n_tokens == 1) {
            s1 = ggml_reshape_2d(ctx0, scores, Pv, n_head_q);
        } else {
            // pool the selection over the TAIL of the segment: the last queries (the
            // ones about to be encoded with the least right-context) dominate what the
            // segment needs to retrieve; full-segment pooling lets bulk filler queries
            // crowd out e.g. a final question's needle pages (measured: needle pages
            // rank top-10 at decode but miss the prefill segment's shared top-k).
            // ablation: LLAMA_LOD_SELPOOL=all restores full-segment pooling
            static const char * pool_env = getenv("LLAMA_LOD_SELPOOL");
            const bool pool_tail = pool_env != nullptr && strcmp(pool_env, "tail") == 0;

            const int64_t n_pool = pool_tail ? std::min<int64_t>(n_tokens, 512) : n_tokens;

            ggml_tensor * sc = scores;
            if (n_pool < n_tokens) {
                sc = ggml_view_3d(ctx0, scores, Pv, n_pool, n_head_q,
                        scores->nb[1], scores->nb[2], (n_tokens - n_pool)*scores->nb[1]);
            }
            s1 = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, 1, n_pool, 1, n_pool, 0, 0); // [Pv, 1, n_head_q]
            s1 = ggml_reshape_2d(ctx0, s1, Pv, n_head_q);
        }

        if (sel_head) {
            ggml_tensor * sg = ggml_reshape_3d(ctx0, s1, Pv, g, n_head_kv);
            ggml_tensor * s2 = ggml_pool_2d(ctx0, sg, GGML_OP_POOL_MAX, 1, g, 1, g, 0, 0); // [Pv, 1, n_head_kv]
            ggml_tensor * sf = ggml_reshape_2d(ctx0, s2, Pv, n_head_kv);
            if (logv) {
                sf = ggml_add(ctx0, sf, ggml_reshape_2d(ctx0, logv, Pv, 1));
            }
            sel = ggml_top_k(ctx0, sf, kP);
        } else {
            ggml_tensor * s2 = ggml_pool_2d(ctx0, s1, GGML_OP_POOL_MAX, 1, n_head_q, 1, n_head_q, 0, 0); // [Pv, 1]
            ggml_tensor * sf = ggml_reshape_1d(ctx0, s2, Pv);
            if (logv) {
                sf = ggml_add(ctx0, sf, logv);
            }
            sel = ggml_top_k(ctx0, sf, kP);
        }
        cb(sel, "lod_sel", il);

        // leaves of the selected pages: gather whole page rows, then slice out each KV head
        // (the fused op reads the leaves in-kernel, no gather needed)
        if (!use_fused) {
        ggml_tensor * k_rows = mctx_cur->get_k_pagerows(ctx0, il, P_full);
        ggml_tensor * v_rows = mctx_cur->get_v_pagerows(ctx0, il, P_full);

        if (sel_head) {
            const int64_t n_embd_k_gqa = n_embd_head_k*n_head_kv;
            const int64_t n_embd_v_gqa = n_embd_head_v*n_head_kv;

            for (int64_t h = 0; h < n_head_kv; ++h) {
                ggml_tensor * sel_h = ggml_view_1d(ctx0, sel, kP, h*kP*sizeof(int32_t));

                ggml_tensor * rk = ggml_get_rows(ctx0, k_rows, sel_h); // F32 [n_embd_k_gqa*ps, kP]
                ggml_tensor * rv = ggml_get_rows(ctx0, v_rows, sel_h);

                ggml_tensor * lk_h = ggml_cont(ctx0, ggml_view_3d(ctx0, rk, n_embd_head_k, ps, kP,
                        n_embd_k_gqa*sizeof(float), rk->nb[1], h*n_embd_head_k*sizeof(float)));
                ggml_tensor * lv_h = ggml_cont(ctx0, ggml_view_3d(ctx0, rv, n_embd_head_v, ps, kP,
                        n_embd_v_gqa*sizeof(float), rv->nb[1], h*n_embd_head_v*sizeof(float)));

                lk_h = ggml_reshape_3d(ctx0, lk_h, n_embd_head_k, NL, 1);
                lv_h = ggml_reshape_3d(ctx0, lv_h, n_embd_head_v, NL, 1);

                lk = lk ? ggml_concat(ctx0, lk, lk_h, 2) : lk_h;
                lv = lv ? ggml_concat(ctx0, lv, lv_h, 2) : lv_h;
            }
        } else {
            lk = ggml_get_rows(ctx0, k_rows, sel); // F32 [n_embd_k_gqa*ps, kP]
            lv = ggml_get_rows(ctx0, v_rows, sel);

            lk = ggml_reshape_4d(ctx0, lk, n_embd_head_k, n_head_kv, ps, kP);
            lv = ggml_reshape_4d(ctx0, lv, n_embd_head_v, n_head_kv, ps, kP);

            lk = ggml_cont(ctx0, ggml_permute(ctx0, lk, 0, 3, 1, 2)); // [D_k, ps, kP, n_head_kv]
            lv = ggml_cont(ctx0, ggml_permute(ctx0, lv, 0, 3, 1, 2));

            lk = ggml_reshape_3d(ctx0, lk, n_embd_head_k, NL, n_head_kv);
            lv = ggml_reshape_3d(ctx0, lv, n_embd_head_v, NL, n_head_kv);
        }
        }
    }

    ggml_tensor * cur;

    if (use_fa && inp->lod_meanidx != nullptr) {
        // mask-direct prefill (ablation LLAMA_LOD_PREFILL=mask): flash attention over
        // [cache span | page-mean columns] with the whole read expressed as a mask -
        // no gathers. Selection is PER 32-QUERY BLOCK (measured: a single shared set
        // cannot serve a whole segment - query granularity is everything, head
        // granularity is nearly free - so pages stay layer/head-shared per block).
        const uint32_t P_cap  = inp->P_cap;
        // span of raw cache columns the mask covers. Capacity (the default) keeps the
        // graph shape constant for every ubatch, which is what graph reuse and the
        // multi-device pipeline both need - a growing span reshapes the graph every
        // few ubatches and forces a reallocation, and under pipeline parallelism that
        // is a full-device stall. The padding costs almost nothing at run time: the
        // columns past the causal frontier are fully masked and the mean tier sits at
        // the front of the axis, so flash-attention's mask scan truncates the dead
        // suffix per query tile. LLAMA_LOD_SPAN=<step> restores the growing span.
        static const char * span_env = getenv("LLAMA_LOD_SPAN");
        const uint32_t span_step = span_env != nullptr ? (uint32_t) std::max(0, atoi(span_env)) : 0;
        const uint32_t S      = span_step > 0
            ? std::min<uint32_t>(GGML_PAD(prev_end + n_tokens, span_step), cparams.n_ctx_seq)
            : cparams.n_ctx_seq;
        inp->mask_S = S;
        const uint32_t P_span = S/ps;
        static const char * qb_env = getenv("LLAMA_LOD_QB");
        const uint32_t QB     = qb_env != nullptr ? std::max(1, atoi(qb_env)) : 32;
        const uint32_t n_blk  = (n_tokens + QB - 1)/QB;
        const uint32_t n_qpad = n_blk*QB;

        const float c0 = (float) prev_end - (float) tail_start; // constant under ps | n_tokens

        // runtime page count as an on-device scalar (stat_fa) or baked constant
        ggml_tensor * prt = nullptr; // [1] = P_full as f32
        if (inp->lod_pmeta != nullptr) {
            prt = inp->lod_pmeta;
        }

        // ---- per-block selection over the page tier (always capacity-sized here;
        // invalid pages are excluded by a local validity vector)
        ggml_tensor * logv_m = logv != nullptr ? logv : inp->sh_mlogv;
        if (logv_m == nullptr) {
            ggml_tensor * ap = ggml_arange(ctx0, 0.5f, (float) P_cap + 0.5f, 1.0f);
            logv_m = ggml_log(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, ap, -1.0f, (float) P_full)));
            inp->sh_mlogv = logv_m;
        }

        ggml_tensor * kp = mctx_cur->get_k_page_sums(ctx0, il, 0, P_cap);
        ggml_tensor * sc = ggml_mul_mat(ctx0, kp, q); // [P_cap, n_tokens, Hq]
        ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
        if (n_qpad > n_tokens) {
            sc = ggml_concat(ctx0, sc,
                    ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, P_cap, n_qpad - n_tokens, n_head_q), -INFINITY), 1);
        }
        ggml_tensor * sb = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, 1, QB, 1, QB, 0, 0);     // [P_cap, n_blk, Hq]
        sb = ggml_cont(ctx0, ggml_permute(ctx0, sb, 0, 2, 1, 3));                            // [P_cap, Hq, n_blk]
        sb = ggml_pool_2d(ctx0, sb, GGML_OP_POOL_MAX, 1, n_head_q, 1, n_head_q, 0, 0);       // [P_cap, 1, n_blk]
        sb = ggml_reshape_2d(ctx0, sb, P_cap, n_blk);
        sb = ggml_add(ctx0, sb, ggml_reshape_2d(ctx0, logv_m, P_cap, 1));

        // selection-only region tier: with thousands of candidate pages the page-level
        // score ranking drowns (measured: works at ~500 pages, dies at ~2000). Prune
        // per block to the top regions (max over 8 child pages - no mean dilution),
        // then pick pages only inside them. Read math is untouched.
        const uint32_t RG = 8;
        if (P_cap % RG == 0 && P_cap/RG >= 64) {
            const uint32_t nreg = P_cap/RG;
            const uint32_t kR   = std::min<uint32_t>(top_regions_l[il], nreg);

            ggml_tensor * rb = ggml_pool_2d(ctx0, ggml_reshape_3d(ctx0, sb, RG, nreg, n_blk),
                    GGML_OP_POOL_MAX, RG, 1, RG, 1, 0, 0);              // [1, nreg, n_blk]
            rb = ggml_reshape_2d(ctx0, ggml_cont(ctx0, rb), nreg, n_blk);
            ggml_tensor * selr = ggml_top_k(ctx0, rb, kR);              // [kR, n_blk]

            ggml_tensor * rk = ggml_cont(ctx0, ggml_repeat(ctx0,
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, 1),
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, nreg, n_blk)));
            rk = ggml_fill_inplace(ctx0, rk, -INFINITY);
            {
                ggml_tensor * zer = ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, kR, n_blk), 0.0f);
                rk = ggml_set_rows(ctx0, rk, zer, ggml_reshape_2d(ctx0, selr, kR, n_blk));
            }
            ggml_tensor * rke = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, rk, 1, nreg, n_blk),
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, RG, nreg, n_blk));
            sb = ggml_add(ctx0, sb, ggml_reshape_2d(ctx0, rke, P_cap, n_blk));
        }

        ggml_tensor * selb = ggml_top_k(ctx0, sb, kP); // [kP, n_blk]

        // ---- token-tier mask: causal base (one formula) + per-block page-kill term
        ggml_tensor * base = inp->sh_mbase;
        if (base == nullptr) {
            ggml_tensor * a_col = ggml_arange(ctx0, 0.0f, (float) S, 1.0f);
            ggml_tensor * a_row = ggml_arange(ctx0, 0.0f, (float) n_tokens, 1.0f);
            base = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, a_row, 1, n_tokens),
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, S, n_tokens));
            base = ggml_sub(ctx0, base, ggml_reshape_2d(ctx0, a_col, S, 1)); // row - col
            if (prt != nullptr) {
                ggml_tensor * off = ggml_scale(ctx0, prt, (float) ps); // P_rt*ps
                base = ggml_add(ctx0, base, ggml_reshape_2d(ctx0, off, 1, 1));
                base = ggml_scale_bias(ctx0, base, 1.0f, c0 + 0.5f);
            } else {
                base = ggml_scale_bias(ctx0, base, 1.0f, (float) tail_start + c0 + 0.5f);
            }
            base = ggml_log(ctx0, ggml_step(ctx0, base)); // 0 visible / -inf beyond
            inp->sh_mbase = base;
        }

        // page-kill column: -inf for complete pages, 0 for the tail region
        ggml_tensor * kill = inp->sh_mkill;
        if (kill == nullptr) {
            ggml_tensor * a_p = ggml_arange(ctx0, 0.5f, (float) P_span + 0.5f, 1.0f); // p + 0.5
            if (prt != nullptr) {
                kill = ggml_sub(ctx0, a_p, ggml_repeat(ctx0, prt, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, P_span)));
            } else {
                kill = ggml_scale_bias(ctx0, a_p, 1.0f, -(float) P_full);
            }
            kill = ggml_log(ctx0, ggml_step(ctx0, kill)); // -inf p < P_rt, 0 past
            inp->sh_mkill = kill;
        }

        ggml_tensor * km = ggml_cont(ctx0, ggml_repeat(ctx0, ggml_reshape_3d(ctx0, kill, 1, P_span, 1),
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, P_span, n_blk)));
        {
            ggml_tensor * zer = ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, kP, n_blk), 0.0f);
            km = ggml_set_rows(ctx0, km, zer, ggml_reshape_2d(ctx0, selb, kP, n_blk)); // un-kill selected
        }
        // expand pages -> token columns, blocks -> token rows
        ggml_tensor * kt = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, km, 1, P_span, n_blk),
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ps, P_span, n_blk));
        kt = ggml_reshape_2d(ctx0, kt, S, n_blk);
        kt = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, kt, S, 1, n_blk),
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, S, QB, n_blk));
        kt = ggml_view_2d(ctx0, ggml_reshape_2d(ctx0, kt, S, n_qpad), S, n_tokens, (size_t) S*sizeof(float), 0);

        ggml_tensor * m_tok = ggml_add(ctx0, base, kt);

        // ---- mean-column mask: logn + validity, selected means silenced per block
        ggml_tensor * ms = ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, P_cap, n_blk), logf((float) ps));
        ms = ggml_add(ctx0, ms, ggml_reshape_3d(ctx0, logv_m, 1, P_cap, 1));
        ms = ggml_cont(ctx0, ms);
        {
            ggml_tensor * neg = ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, kP, n_blk), -INFINITY);
            ms = ggml_set_rows(ctx0, ms, neg, ggml_reshape_2d(ctx0, selb, kP, n_blk));
        }
        ggml_tensor * mst = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, ms, P_cap, 1, n_blk),
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, P_cap, QB, n_blk));
        mst = ggml_view_2d(ctx0, ggml_reshape_2d(ctx0, mst, P_cap, n_qpad), P_cap, n_tokens, (size_t) P_cap*sizeof(float), 0);

        // ---- K/V: resident mean columns first, cache span last (+ zeroed pad rows
        // reserved inside the mean tensors - all-zero quantized blocks decode to 0.0).
        // fattn's mask scan can only truncate a fully-masked KV suffix, so the
        // causally-dead token tail must sit at the end of the axis; a mean tier at the
        // end would keep every query tile pinned to the full span. The concat runs on
        // contiguous 2D row tensors: that is the byte-copy path, which works for
        // quantized types too; the result is re-viewed in fattn shape.
        const uint32_t N_set  = S + P_cap;
        const uint32_t n_fpad = GGML_PAD(N_set, 256) - N_set;

        ggml_tensor * kr = ggml_concat(ctx0, mctx_cur->get_k_meanrows(ctx0, il, P_cap + n_fpad), mctx_cur->get_k_tokrows(ctx0, il, S), 1);
        ggml_tensor * vr = ggml_concat(ctx0, mctx_cur->get_v_meanrows(ctx0, il, P_cap + n_fpad), mctx_cur->get_v_tokrows(ctx0, il, S), 1);

        ggml_tensor * kk = ggml_view_3d(ctx0, kr, n_embd_head_k, N_set + n_fpad, n_head_kv,
                kr->nb[1], ggml_row_size(kr->type, n_embd_head_k), 0);
        ggml_tensor * vv = ggml_view_3d(ctx0, vr, n_embd_head_v, N_set + n_fpad, n_head_kv,
                vr->nb[1], ggml_row_size(vr->type, n_embd_head_v), 0);

        ggml_tensor * mask = mst;
        if (n_fpad > 0) {
            mask = ggml_concat(ctx0, mask, ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_fpad, n_tokens), -INFINITY), 0);
        }
        mask = ggml_concat(ctx0, mask, m_tok, 0); // [P_cap + n_fpad + S, n_tokens]
        mask = ggml_cast(ctx0, ggml_reshape_3d(ctx0, mask, N_set + n_fpad, n_tokens, 1), GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, kk, vv, mask, kq_scale, 0.0f, 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else if (use_fa) {
        // one flash-attention call over the compact [leaf | exact | summary] KV set
        // (quantized caches read the exact tier through a dequantizing gather)
        ggml_tensor * ke;
        ggml_tensor * ve;
        if (inp->lod_arange) {
            ggml_tensor * ker = ggml_get_rows(ctx0, mctx_cur->get_k_tokrows(ctx0, il, cparams.n_ctx_seq), inp->lod_arange);
            ggml_tensor * ver = ggml_get_rows(ctx0, mctx_cur->get_v_tokrows(ctx0, il, cparams.n_ctx_seq), inp->lod_arange);

            ke = ggml_cast(ctx0, ggml_view_3d(ctx0, ker, n_embd_head_k, n_exact, n_head_kv, ker->nb[1], n_embd_head_k*sizeof(float), 0), GGML_TYPE_F16);
            ve = ggml_cast(ctx0, ggml_view_3d(ctx0, ver, n_embd_head_v, n_exact, n_head_kv, ver->nb[1], n_embd_head_v*sizeof(float), 0), GGML_TYPE_F16);
        } else {
            ke = mctx_cur->get_k_range(ctx0, il, tail_start, n_exact);
            ve = mctx_cur->get_v_range(ctx0, il, tail_start, n_exact);
        }

        ggml_tensor * kk;
        ggml_tensor * vv;
        ggml_tensor * mask;

        if (kP > 0) {
            ggml_tensor * ks = ggml_cast(ctx0, ggml_scale(ctx0, ggml_cont(ctx0, mctx_cur->get_k_page_sums(ctx0, il, 0, Pv)), 1.0f/ps), GGML_TYPE_F16);
            ggml_tensor * vs = ggml_cast(ctx0, ggml_scale(ctx0, ggml_cont(ctx0, mctx_cur->get_v_page_sums(ctx0, il, 0, Pv)), 1.0f/ps), GGML_TYPE_F16);

            kk = ggml_concat(ctx0, ggml_concat(ctx0, ggml_cast(ctx0, lk, GGML_TYPE_F16), ke, 1), ks, 1); // [D_k, NL+n_exact+P_full, n_head_kv]
            vv = ggml_concat(ctx0, ggml_concat(ctx0, ggml_cast(ctx0, lv, GGML_TYPE_F16), ve, 1), vs, 1);

            // silence the selected summaries in a private copy of the mask segment
            // (the mask tensor is shared by every LoD layer, it must not be scattered in place)
            ggml_tensor * m_sum2 = ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Pv, n_tokens), logf((float) ps));
            if (logv) {
                m_sum2 = ggml_add(ctx0, m_sum2, ggml_reshape_2d(ctx0, logv, Pv, 1)); // -inf past the runtime page count
            }
            ggml_tensor * m_sum = ggml_reshape_4d(ctx0, m_sum2, 1, Pv, n_tokens, 1);

            ggml_tensor * neg = ggml_fill_inplace(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, kP, n_tokens, 1), -INFINITY);

            m_sum = ggml_set_rows(ctx0, m_sum, neg, ggml_reshape_3d(ctx0, sel, kP, 1, 1));
            m_sum = ggml_reshape_3d(ctx0, m_sum, Pv, n_tokens, 1);

            ggml_tensor * mpre2 = kp_uniform ? inp->sh_mpre : nullptr;
            if (mpre2 == nullptr) {
                mpre2 = ggml_concat(ctx0, m_leaf, m_band, 0);
                if (kp_uniform) {
                    inp->sh_mpre = mpre2;
                }
            }
            ggml_tensor * m_pre = ggml_reshape_3d(ctx0, mpre2, NL + n_exact, n_tokens, 1);

            mask = ggml_concat(ctx0, m_pre, m_sum, 0);
        } else {
            kk   = ke;
            vv   = ve;
            mask = ggml_reshape_3d(ctx0, m_band, n_exact, n_tokens, 1);
        }

        // the fattn kernels want the KV length padded to FATTN_KQ_STRIDE; padded columns
        // are fully masked, so their blocks are skipped
        const uint32_t N_set  = kP > 0 ? NL + n_exact + Pv : n_exact;
        const uint32_t n_fpad = GGML_PAD(N_set, 256) - N_set;

        if (n_fpad > 0) {
            kk = ggml_concat(ctx0, kk, ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, kk->type, n_embd_head_k, n_fpad, n_head_kv), 0.0f), 1);
            vv = ggml_concat(ctx0, vv, ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, vv->type, n_embd_head_v, n_fpad, n_head_kv), 0.0f), 1);

            mask = ggml_concat(ctx0, mask, ggml_fill_inplace(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_fpad, n_tokens, 1), -INFINITY), 0);
        }

        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, kk, vv, mask, kq_scale, 0.0f, 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else if (use_fused && kP > 0) {
        // single-kernel read over all tiers; every view is full-capacity and the geometry
        // travels through lod_meta, so the graph shape never changes - full reuse.
        // the op also folds the current tokens into the page sums (reading them from the
        // cache columns the write above just filled), so update_sums is skipped entirely
        ggml_tensor * kf = mctx_cur->get_k_range(ctx0, il, 0, cparams.n_ctx_seq);
        ggml_tensor * vf = mctx_cur->get_v_range(ctx0, il, 0, cparams.n_ctx_seq);

        cur = ggml_lod_attn(ctx0, q, kf, vf,
                mctx_cur->get_k_page_sums(ctx0, il, 0, cparams.n_ctx_seq/ps),
                mctx_cur->get_v_page_sums(ctx0, il, 0, cparams.n_ctx_seq/ps),
                nullptr, inp->lod_meta, ps, kq_scale, top_pages_l[il],
                cparams.lod_sel_head && n_head_kv > 1);
        cb(cur, "lod_fused", il);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * k_exact;
        ggml_tensor * v_exact_f;
        if (q_cache) {
            ggml_tensor * ker = ggml_get_rows(ctx0, mctx_cur->get_k_tokrows(ctx0, il, cparams.n_ctx_seq), inp->lod_arange);
            ggml_tensor * ver = ggml_get_rows(ctx0, mctx_cur->get_v_tokrows(ctx0, il, cparams.n_ctx_seq), inp->lod_arange);

            k_exact   = ggml_view_3d(ctx0, ker, n_embd_head_k, n_exact, n_head_kv, ker->nb[1], n_embd_head_k*sizeof(float), 0);
            v_exact_f = ggml_view_3d(ctx0, ver, n_embd_head_v, n_exact, n_head_kv, ver->nb[1], n_embd_head_v*sizeof(float), 0);
        } else {
            k_exact   = mctx_cur->get_k_range(ctx0, il, tail_start, n_exact);
            v_exact_f = mctx_cur->get_v_range(ctx0, il, tail_start, n_exact);
        }

        ggml_tensor * lg_exact = ggml_mul_mat(ctx0, k_exact, q); // [n_exact, n_tokens, n_head_q]
        ggml_mul_mat_set_prec(lg_exact, GGML_PREC_F32);
        cb(lg_exact, "lod_lg_exact", il);

        ggml_tensor * logits = lg_exact;
        ggml_tensor * lvT    = nullptr;

        if (kP > 0) {
            ggml_tensor * lg_leaf = ggml_mul_mat(ctx0, lk, q); // [NL, n_tokens, n_head_q]
            ggml_mul_mat_set_prec(lg_leaf, GGML_PREC_F32);
            cb(lg_leaf, "lod_lg_leaf", il);

            const bool    sel_head = sel->ne[1] > 1;
            const int64_t g        = n_head_q / n_head_kv;

            // refinement: a selected page contributes its leaves, so its summary logit goes silent
            ggml_tensor * s_sil;
            if (sel_head) {
                ggml_tensor * neg = ggml_fill_inplace(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, kP, n_tokens*g, n_head_kv), -INFINITY);

                s_sil = ggml_set_rows(ctx0,
                        ggml_reshape_4d(ctx0, scores, 1, P_full, n_tokens*g, n_head_kv),
                        neg,
                        ggml_reshape_3d(ctx0, sel, kP, 1, n_head_kv));
            } else {
                ggml_tensor * neg = ggml_fill_inplace(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, kP, n_tokens, n_head_q), -INFINITY);

                s_sil = ggml_set_rows(ctx0,
                        ggml_reshape_4d(ctx0, scores, 1, P_full, n_tokens, n_head_q),
                        neg,
                        ggml_reshape_3d(ctx0, sel, kP, 1, 1));
            }
            s_sil = ggml_reshape_3d(ctx0, s_sil, P_full, n_tokens, n_head_q);

            logits = ggml_concat(ctx0, ggml_concat(ctx0, lg_leaf, lg_exact, 0), s_sil, 0); // [NL + n_exact + P_full, n_tokens, n_head_q]

            lvT = ggml_cont(ctx0, ggml_permute(ctx0, lv, 1, 0, 2, 3)); // [NL, D_v, n_head_kv]
        }

        // one softmax over every tier - the denominator is complete for any selection
        ggml_tensor * m_full = m_band;
        if (kP > 0) {
            m_full = ggml_concat(ctx0, ggml_concat(ctx0, m_leaf, m_band, 0),
                    ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, P_full, n_tokens), logf((float) ps)), 0);
        }
        m_full = ggml_reshape_3d(ctx0, m_full, m_full->ne[0], n_tokens, 1);

        ggml_tensor * probs = ggml_soft_max_ext(ctx0, logits, m_full, kq_scale, 0.0f);
        cb(probs, "lod_probs", il);

        ggml_tensor * veT = ggml_cont(ctx0, ggml_permute(ctx0, v_exact_f, 1, 0, 2, 3)); // [n_exact, D_v, n_head_kv]

        if (kP > 0) {
            // contiguous copies: the CUDA mul_mat_vec path assumes naturally-strided src1
            ggml_tensor * p_leaf  = ggml_cont(ctx0, ggml_view_3d(ctx0, probs, NL,      n_tokens, n_head_q, probs->nb[1], probs->nb[2], 0));
            ggml_tensor * p_exact = ggml_cont(ctx0, ggml_view_3d(ctx0, probs, n_exact, n_tokens, n_head_q, probs->nb[1], probs->nb[2], NL*sizeof(float)));
            ggml_tensor * p_sum   = ggml_cont(ctx0, ggml_view_3d(ctx0, probs, P_full,  n_tokens, n_head_q, probs->nb[1], probs->nb[2], (NL + n_exact)*(size_t) sizeof(float)));

            ggml_tensor * vp  = mctx_cur->get_v_page_sums(ctx0, il, 0, P_full);
            ggml_tensor * vpT = ggml_cont(ctx0, ggml_permute(ctx0, vp, 1, 0, 2, 3)); // [P_full, D_v, n_head_kv]

            cur = ggml_mul_mat(ctx0, lvT, p_leaf);
            cur = ggml_add(ctx0, cur, ggml_mul_mat(ctx0, veT, p_exact));
            cur = ggml_add(ctx0, cur, ggml_scale(ctx0, ggml_mul_mat(ctx0, vpT, p_sum), 1.0f/ps));
        } else {
            cur = ggml_mul_mat(ctx0, veT, probs);
        }

        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    }

    ggml_build_forward_expand(gf, cur);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    int32_t logit_row_idx = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            seq_to_logit_row[seq_id] = logit_row_idx;
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ nullptr,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
