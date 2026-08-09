// test-lod-attention.cpp: LoD (level-of-detail) sparse attention read composed from stock ggml ops.
//
// The read splits the keys of one attention layer into three tiers:
//   leaves    the raw K/V of the top-k selected pages (page = ps contiguous tokens)
//   exact     the raw tail (tokens not yet covered by a full page) + the current queries, causal
//   summaries one mean K/V per page, carrying a log(page_size) size bias
// A selected page contributes its leaves and its summary is silenced with -INF
// (refinement, not subtraction). All three tiers share ONE softmax, so the
// denominator is complete for any selection.
//
// Graph idioms under test (they must survive into the llama.cpp integration):
//   - page gather without index math: view K as page rows [D*ps, P, Hkv] + ggml_get_rows
//   - per-KV-head batched selection: ggml_top_k on [P, cols, Hkv]
//   - summary silencing: ggml_set_rows scatter of -INF rows into the mask
//   - one ggml_soft_max_ext over concatenated logits with an additive mask
//
// Checks, on every available backend:
//   - kP == P (every page expanded) must equal dense causal attention
//   - kP <  P must equal a scalar double-precision reference of the refinement formula
//   - the graph's top-k page set must equal the reference's (as a set; order is free)

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <set>
#include <vector>

struct lod_case {
    int64_t D;      // head dim
    int64_t Hkv;    // KV heads
    int64_t g;      // GQA group (query heads per KV head)
    int64_t ps;     // page size
    int64_t P;      // full pages in the index
    int64_t n_tail; // raw tokens after the last full page
    int64_t nq;     // current query tokens
    int64_t kP;     // pages selected
    bool    shared; // one selection shared by all queries (prefill) vs per query (decode)
    ggml_type tkv;  // storage type of the paged K/V (leaves dequantize through get_rows)
};

static const lod_case CASES[] = {
    // D  Hkv  g  ps   P  tail nq  kP  shared  type
    {  32,  2, 2,  8,  6,  5,  1,  2, false, GGML_TYPE_F32  }, // decode, partial
    {  32,  2, 2,  8,  6,  5,  1,  6, false, GGML_TYPE_F32  }, // decode, full expansion == dense
    {  64,  1, 4, 16,  8,  0,  4,  3, false, GGML_TYPE_F32  }, // MQA, multi-token decode, no tail
    {  64,  1, 4, 16,  8,  0,  4,  8, false, GGML_TYPE_F32  },
    {  64,  2, 2, 16,  8,  7, 32,  4, true,  GGML_TYPE_F32  }, // prefill block, shared selection
    {  64,  2, 2, 16,  8,  7, 32,  8, true,  GGML_TYPE_F32  },
    { 128,  2, 8, 16, 10,  3,  5,  2, false, GGML_TYPE_F32  }, // gemma4-like GQA=8
    { 128,  2, 8, 16, 10,  3,  5, 10, false, GGML_TYPE_F32  },
    {  64,  2, 2, 16,  8,  5,  1,  3, false, GGML_TYPE_Q8_0 }, // quantized leaves (kv cache quant reuse)
    {  64,  2, 2, 16,  8,  5,  1,  8, false, GGML_TYPE_Q8_0 },
    {  64,  2, 2, 16,  8,  7, 32,  4, true,  GGML_TYPE_Q4_0 },
    {  64,  2, 2, 16,  8,  7, 32,  8, true,  GGML_TYPE_Q4_0 },
};

static uint64_t g_rng;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

static std::vector<float> rand_vec(int64_t n) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) v[i] = frand();
    return v;
}

struct ref_out {
    std::vector<double> out;            // [D * cols * Hkv], d fastest
    std::vector<std::set<int64_t>> sel; // per (kv, col), or per kv when shared
};

// scalar double reference of the refinement read
static ref_out ref_lod(const lod_case & c,
                       const std::vector<float> & q,        // [D][col][kv], d fastest
                       const std::vector<float> & k_idx,    // [D][t][kv], t = p*ps+j
                       const std::vector<float> & v_idx,
                       const std::vector<float> & k_exact,  // [D][i][kv]
                       const std::vector<float> & v_exact,
                       const std::vector<float> & k_page,   // [D][p][kv]
                       const std::vector<float> & v_page,
                       float scale) {
    const int64_t D = c.D, cols = c.g * c.nq, n_exact = c.n_tail + c.nq;
    ref_out r;
    r.out.assign(D * cols * c.Hkv, 0.0);

    auto qv = [&](int64_t d, int64_t col, int64_t kv) { return (double) q[d + D*(col + cols*kv)]; };
    auto kx = [&](int64_t d, int64_t t,   int64_t kv) { return (double) k_idx[d + D*(t + c.ps*c.P*kv)]; };
    auto vx = [&](int64_t d, int64_t t,   int64_t kv) { return (double) v_idx[d + D*(t + c.ps*c.P*kv)]; };
    auto ke = [&](int64_t d, int64_t i,   int64_t kv) { return (double) k_exact[d + D*(i + n_exact*kv)]; };
    auto ve = [&](int64_t d, int64_t i,   int64_t kv) { return (double) v_exact[d + D*(i + n_exact*kv)]; };
    auto kp = [&](int64_t d, int64_t p,   int64_t kv) { return (double) k_page[d + D*(p + c.P*kv)]; };
    auto vp = [&](int64_t d, int64_t p,   int64_t kv) { return (double) v_page[d + D*(p + c.P*kv)]; };

    auto page_score = [&](int64_t col, int64_t kv, int64_t p) {
        double s = 0.0;
        for (int64_t d = 0; d < D; d++) {
            s += qv(d, col, kv) * kp(d, p, kv);
        }
        return s;
    };
    auto topk = [&](const std::vector<double> & s) {
        std::vector<int64_t> idx(c.P);
        for (int64_t p = 0; p < c.P; p++) idx[p] = p;
        std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return s[a] > s[b]; });
        return std::set<int64_t>(idx.begin(), idx.begin() + c.kP);
    };

    std::vector<std::set<int64_t>> sel(c.Hkv * (c.shared ? 1 : cols));
    for (int64_t kv = 0; kv < c.Hkv; kv++) {
        if (c.shared) {
            std::vector<double> pooled(c.P, -1e300);
            for (int64_t col = 0; col < cols; col++) {
                for (int64_t p = 0; p < c.P; p++) {
                    pooled[p] = std::max(pooled[p], page_score(col, kv, p));
                }
            }
            sel[kv] = topk(pooled);
        } else {
            for (int64_t col = 0; col < cols; col++) {
                std::vector<double> s(c.P);
                for (int64_t p = 0; p < c.P; p++) s[p] = page_score(col, kv, p);
                sel[kv * cols + col] = topk(s);
            }
        }
    }
    r.sel = sel;

    for (int64_t kv = 0; kv < c.Hkv; kv++) {
        for (int64_t col = 0; col < cols; col++) {
            const int64_t qi = col / c.g; // col = j + g*qi
            const std::set<int64_t> & s = sel[c.shared ? kv : kv * cols + col];

            std::vector<double> lg;
            std::vector<char>    tier;
            std::vector<int64_t> ref_i;

            for (int64_t p : s) {
                for (int64_t j = 0; j < c.ps; j++) {
                    double l = 0.0;
                    for (int64_t d = 0; d < D; d++) l += qv(d, col, kv) * kx(d, p*c.ps + j, kv);
                    lg.push_back(l * scale); tier.push_back('l'); ref_i.push_back(p*c.ps + j);
                }
            }
            for (int64_t i = 0; i < n_exact; i++) {
                const bool vis = i < c.n_tail || (i - c.n_tail) <= qi;
                if (!vis) continue;
                double l = 0.0;
                for (int64_t d = 0; d < D; d++) l += qv(d, col, kv) * ke(d, i, kv);
                lg.push_back(l * scale); tier.push_back('e'); ref_i.push_back(i);
            }
            for (int64_t p = 0; p < c.P; p++) {
                if (s.count(p)) continue;
                lg.push_back(page_score(col, kv, p) * scale + log((double) c.ps));
                tier.push_back('p'); ref_i.push_back(p);
            }

            double mx = -1e300;
            for (double l : lg) mx = std::max(mx, l);
            double den = 0.0;
            std::vector<double> w(lg.size());
            for (size_t i = 0; i < lg.size(); i++) { w[i] = exp(lg[i] - mx); den += w[i]; }

            for (int64_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (size_t i = 0; i < lg.size(); i++) {
                    double v = tier[i] == 'l' ? vx(d, ref_i[i], kv)
                             : tier[i] == 'e' ? ve(d, ref_i[i], kv)
                             :                  vp(d, ref_i[i], kv);
                    acc += w[i] * v;
                }
                r.out[d + D*(col + cols*kv)] = acc / den;
            }
        }
    }
    return r;
}

// dense causal reference over every real token (index + exact), no hierarchy
static std::vector<double> ref_dense(const lod_case & c,
                                     const std::vector<float> & q,
                                     const std::vector<float> & k_idx,
                                     const std::vector<float> & v_idx,
                                     const std::vector<float> & k_exact,
                                     const std::vector<float> & v_exact,
                                     float scale) {
    const int64_t D = c.D, cols = c.g * c.nq, n_exact = c.n_tail + c.nq, n_idx = c.P * c.ps;
    std::vector<double> out(D * cols * c.Hkv, 0.0);
    for (int64_t kv = 0; kv < c.Hkv; kv++) {
        for (int64_t col = 0; col < cols; col++) {
            const int64_t qi = col / c.g;
            std::vector<double> lg;
            std::vector<int64_t> src;
            for (int64_t t = 0; t < n_idx; t++) {
                double l = 0.0;
                for (int64_t d = 0; d < D; d++) {
                    l += (double) q[d + D*(col + cols*kv)] * (double) k_idx[d + D*(t + n_idx*kv)];
                }
                lg.push_back(l * scale); src.push_back(t);
            }
            for (int64_t i = 0; i < n_exact; i++) {
                if (!(i < c.n_tail || (i - c.n_tail) <= qi)) continue;
                double l = 0.0;
                for (int64_t d = 0; d < D; d++) {
                    l += (double) q[d + D*(col + cols*kv)] * (double) k_exact[d + D*(i + n_exact*kv)];
                }
                lg.push_back(l * scale); src.push_back(n_idx + i);
            }
            double mx = -1e300;
            for (double l : lg) mx = std::max(mx, l);
            double den = 0.0;
            std::vector<double> w(lg.size());
            for (size_t i = 0; i < lg.size(); i++) { w[i] = exp(lg[i] - mx); den += w[i]; }
            for (int64_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (size_t i = 0; i < lg.size(); i++) {
                    acc += w[i] * (src[i] < n_idx ? (double) v_idx[d + D*(src[i] + n_idx*kv)]
                                                  : (double) v_exact[d + D*(src[i] - n_idx + n_exact*kv)]);
                }
                out[d + D*(col + cols*kv)] = acc / den;
            }
        }
    }
    return out;
}

static double nmse(const float * y, const double * ref, int64_t n) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        num += ((double) y[i] - ref[i]) * ((double) y[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return num / (den + 1e-30);
}

// 0 = pass, 1 = fail, -1 = backend does not support an op (reported, not a failure)
static int run_case(const lod_case & c, ggml_backend_t backend) {
    const int64_t D = c.D, Hkv = c.Hkv, cols = c.g * c.nq;
    const int64_t n_idx = c.P * c.ps, n_exact = c.n_tail + c.nq;
    const int64_t NL = c.ps * c.kP; // leaf logits per row
    const float scale = 1.0f / sqrtf((float) D);

    // deterministic host data, identical on every backend
    g_rng = 0x243F6A8885A308D3ULL ^ (uint64_t) (D*1000003 + c.P*10007 + c.kP*101 + c.shared + c.tkv);
    const std::vector<float> qh  = rand_vec(D * cols * Hkv);
    std::vector<float> kih = rand_vec(D * n_idx * Hkv);
    std::vector<float> vih = rand_vec(D * n_idx * Hkv);
    const std::vector<float> keh = rand_vec(D * n_exact * Hkv);
    const std::vector<float> veh = rand_vec(D * n_exact * Hkv);

    // quantized leaves: the reference reads what the cache stores, i.e. the dequantized values
    std::vector<uint8_t> kiq, viq;
    if (c.tkv != GGML_TYPE_F32) {
        const int64_t n_rows_q = n_idx * Hkv;
        kiq.resize(ggml_row_size(c.tkv, D) * n_rows_q);
        viq.resize(ggml_row_size(c.tkv, D) * n_rows_q);
        ggml_quantize_chunk(c.tkv, kih.data(), kiq.data(), 0, n_rows_q, D, nullptr);
        ggml_quantize_chunk(c.tkv, vih.data(), viq.data(), 0, n_rows_q, D, nullptr);
        const auto * traits = ggml_get_type_traits(c.tkv);
        traits->to_float(kiq.data(), kih.data(), D * n_rows_q);
        traits->to_float(viq.data(), vih.data(), D * n_rows_q);
    }

    // page summaries = per-page mean (the integration will maintain these incrementally)
    std::vector<float> kph(D * c.P * Hkv), vph(D * c.P * Hkv);
    for (int64_t kv = 0; kv < Hkv; kv++) {
        for (int64_t p = 0; p < c.P; p++) {
            for (int64_t d = 0; d < D; d++) {
                double sk = 0.0, sv = 0.0;
                for (int64_t j = 0; j < c.ps; j++) {
                    sk += kih[d + D*(p*c.ps + j + n_idx*kv)];
                    sv += vih[d + D*(p*c.ps + j + n_idx*kv)];
                }
                kph[d + D*(p + c.P*kv)] = (float) (sk / c.ps);
                vph[d + D*(p + c.P*kv)] = (float) (sv / c.ps);
            }
        }
    }

    // additive masks: exact tier is causal, leaf tier all live, summary tier carries log(n)
    std::vector<float> mask_exact_h(n_exact * cols * Hkv);
    for (int64_t kv = 0; kv < Hkv; kv++) {
        for (int64_t col = 0; col < cols; col++) {
            const int64_t qi = col / c.g;
            for (int64_t i = 0; i < n_exact; i++) {
                const bool vis = i < c.n_tail || (i - c.n_tail) <= qi;
                mask_exact_h[i + n_exact*(col + cols*kv)] = vis ? 0.0f : -INFINITY;
            }
        }
    }
    const std::vector<float> mask_leaf_h(NL * cols * Hkv, 0.0f);
    const std::vector<float> mask_sum_h(c.P * cols * Hkv, logf((float) c.ps));
    const std::vector<float> neg_inf_h(c.kP * cols * Hkv, -INFINITY);

    // ---- graph ----
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, cols, Hkv);
    struct ggml_tensor * k_idx   = ggml_new_tensor_3d(ctx, c.tkv, D, n_idx, Hkv);
    struct ggml_tensor * v_idx   = ggml_new_tensor_3d(ctx, c.tkv, D, n_idx, Hkv);
    struct ggml_tensor * k_exact = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_exact, Hkv);
    struct ggml_tensor * v_exact = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_exact, Hkv);
    struct ggml_tensor * k_page  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.P, Hkv);
    struct ggml_tensor * v_page  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.P, Hkv);

    struct ggml_tensor * mask_leaf     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NL, 1, cols, Hkv);
    struct ggml_tensor * mask_exact    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_exact, 1, cols, Hkv);
    struct ggml_tensor * mask_sum_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.P, cols, Hkv);
    struct ggml_tensor * neg_inf       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.kP, cols, Hkv);

    // selection scores: [P, cols, Hkv]; scale is rank-preserving so top_k reads them unscaled
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k_page, q);

    struct ggml_tensor * sel;         // [kP, cols, Hkv] or [kP, Hkv]
    struct ggml_tensor * sel_gather;  // shaped for get_rows: [*, Hkv]
    struct ggml_tensor * sel_scatter; // shaped for set_rows
    if (c.shared) {
        struct ggml_tensor * pooled = ggml_pool_2d(ctx, scores, GGML_OP_POOL_MAX, 1, cols, 1, cols, 0, 0); // [P, 1, Hkv]
        sel         = ggml_top_k(ctx, ggml_reshape_2d(ctx, pooled, c.P, Hkv), c.kP);   // [kP, Hkv]
        sel_gather  = sel;
        sel_scatter = ggml_reshape_3d(ctx, sel, c.kP, 1, Hkv);
    } else {
        sel         = ggml_top_k(ctx, scores, c.kP);                                   // [kP, cols, Hkv]
        sel_gather  = ggml_reshape_2d(ctx, sel, c.kP * cols, Hkv);
        sel_scatter = sel;
    }

    // leaf gather: index K/V viewed as page rows [D*ps, P, Hkv]; a page row stays block-aligned for quantized types
    struct ggml_tensor * k_rows = ggml_view_3d(ctx, k_idx, D*c.ps, c.P, Hkv, ggml_row_size(c.tkv, D*c.ps), k_idx->nb[2], 0);
    struct ggml_tensor * v_rows = ggml_view_3d(ctx, v_idx, D*c.ps, c.P, Hkv, ggml_row_size(c.tkv, D*c.ps), v_idx->nb[2], 0);

    struct ggml_tensor * lk = ggml_get_rows(ctx, k_rows, sel_gather); // [D*ps, kP(*cols), Hkv]
    struct ggml_tensor * lv = ggml_get_rows(ctx, v_rows, sel_gather);

    struct ggml_tensor * logits_leaf;
    struct ggml_tensor * lvT;
    if (c.shared) {
        lk = ggml_reshape_3d(ctx, lk, D, NL, Hkv);
        lv = ggml_reshape_3d(ctx, lv, D, NL, Hkv);
        logits_leaf = ggml_mul_mat(ctx, lk, q);                                     // [NL, cols, Hkv]
        lvT = ggml_cont(ctx, ggml_permute(ctx, lv, 1, 0, 2, 3));                    // [NL, D, Hkv]
    } else {
        lk = ggml_reshape_4d(ctx, lk, D, NL, cols, Hkv);
        lv = ggml_reshape_4d(ctx, lv, D, NL, cols, Hkv);
        struct ggml_tensor * q_cols = ggml_reshape_4d(ctx, q, D, 1, cols, Hkv);
        logits_leaf = ggml_mul_mat(ctx, lk, q_cols);                                // [NL, 1, cols, Hkv]
        lvT = ggml_cont(ctx, ggml_permute(ctx, lv, 1, 0, 2, 3));                    // [NL, D, cols, Hkv]
    }

    struct ggml_tensor * logits_exact = ggml_mul_mat(ctx, k_exact, q);              // [n_exact, cols, Hkv]

    // silence selected summaries: scatter -INF rows into the log(n) mask
    struct ggml_tensor * mask_sum = ggml_set_rows(ctx, mask_sum_base, neg_inf, sel_scatter); // [1, P, cols, Hkv]

    // concat tiers into one logit row: [leaf | exact | summary]
    struct ggml_tensor * logits;
    struct ggml_tensor * mask;
    if (c.shared) {
        logits = ggml_concat(ctx, ggml_concat(ctx, logits_leaf, logits_exact, 0),
                             scores, 0);                                            // [N, cols, Hkv]
        mask   = ggml_concat(ctx, ggml_concat(ctx,
                     ggml_reshape_3d(ctx, mask_leaf, NL, cols, Hkv),
                     ggml_reshape_3d(ctx, mask_exact, n_exact, cols, Hkv), 0),
                     ggml_reshape_3d(ctx, mask_sum, c.P, cols, Hkv), 0);
    } else {
        logits = ggml_concat(ctx, ggml_concat(ctx, logits_leaf,
                     ggml_reshape_4d(ctx, logits_exact, n_exact, 1, cols, Hkv), 0),
                     ggml_reshape_4d(ctx, scores, c.P, 1, cols, Hkv), 0);           // [N, 1, cols, Hkv]
        mask   = ggml_concat(ctx, ggml_concat(ctx, mask_leaf, mask_exact, 0),
                     ggml_reshape_4d(ctx, mask_sum, c.P, 1, cols, Hkv), 0);
    }

    struct ggml_tensor * probs = ggml_soft_max_ext(ctx, logits, mask, scale, 0.0f);

    // split the probability row back into tiers and contract each against its values
    struct ggml_tensor * out;
    if (c.shared) {
        struct ggml_tensor * p_leaf  = ggml_view_3d(ctx, probs, NL,      cols, Hkv, probs->nb[1], probs->nb[2], 0);
        struct ggml_tensor * p_exact = ggml_view_3d(ctx, probs, n_exact, cols, Hkv, probs->nb[1], probs->nb[2], NL*sizeof(float));
        struct ggml_tensor * p_sum   = ggml_view_3d(ctx, probs, c.P,     cols, Hkv, probs->nb[1], probs->nb[2], (NL + n_exact)*sizeof(float));

        struct ggml_tensor * veT = ggml_cont(ctx, ggml_permute(ctx, v_exact, 1, 0, 2, 3)); // [n_exact, D, Hkv]
        struct ggml_tensor * vpT = ggml_cont(ctx, ggml_permute(ctx, v_page,  1, 0, 2, 3)); // [P, D, Hkv]

        out = ggml_add(ctx, ggml_add(ctx,
                  ggml_mul_mat(ctx, lvT, p_leaf),
                  ggml_mul_mat(ctx, veT, p_exact)),
                  ggml_mul_mat(ctx, vpT, p_sum));                                   // [D, cols, Hkv]
    } else {
        struct ggml_tensor * p_leaf  = ggml_view_4d(ctx, probs, NL, 1, cols, Hkv, probs->nb[1], probs->nb[2], probs->nb[3], 0);
        struct ggml_tensor * p_exact = ggml_view_3d(ctx, probs, n_exact, cols, Hkv, probs->nb[2], probs->nb[3], NL*sizeof(float));
        struct ggml_tensor * p_sum   = ggml_view_3d(ctx, probs, c.P,     cols, Hkv, probs->nb[2], probs->nb[3], (NL + n_exact)*sizeof(float));

        struct ggml_tensor * veT = ggml_cont(ctx, ggml_permute(ctx, v_exact, 1, 0, 2, 3));
        struct ggml_tensor * vpT = ggml_cont(ctx, ggml_permute(ctx, v_page,  1, 0, 2, 3));

        struct ggml_tensor * o_leaf = ggml_mul_mat(ctx, lvT, p_leaf);               // [D, 1, cols, Hkv]
        out = ggml_add(ctx, ggml_add(ctx,
                  ggml_reshape_3d(ctx, o_leaf, D, cols, Hkv),
                  ggml_mul_mat(ctx, veT, p_exact)),
                  ggml_mul_mat(ctx, vpT, p_sum));                                   // [D, cols, Hkv]
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, sel);

    // report ops this backend cannot run; not a numeric failure
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node)) {
            printf("  unsupported op: %s (%s)\n", ggml_op_desc(node), node->name);
            ggml_free(ctx);
            return -1;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) {
        printf("  buffer allocation failed\n");
        ggml_free(ctx);
        return 1;
    }

    auto upload = [](struct ggml_tensor * t, const std::vector<float> & v) {
        GGML_ASSERT((int64_t) v.size() == ggml_nelements(t));
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };
    upload(q, qh);
    if (c.tkv == GGML_TYPE_F32) {
        upload(k_idx, kih); upload(v_idx, vih);
    } else {
        ggml_backend_tensor_set(k_idx, kiq.data(), 0, kiq.size());
        ggml_backend_tensor_set(v_idx, viq.data(), 0, viq.size());
    }
    upload(k_exact, keh); upload(v_exact, veh);
    upload(k_page, kph); upload(v_page, vph);
    upload(mask_leaf, mask_leaf_h); upload(mask_exact, mask_exact_h);
    upload(mask_sum_base, mask_sum_h); upload(neg_inf, neg_inf_h);

    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed: %d\n", (int) st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> y(ggml_nelements(out));
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));
    std::vector<int32_t> sel_h(ggml_nelements(sel));
    ggml_backend_tensor_get(sel, sel_h.data(), 0, sel_h.size() * sizeof(int32_t));

    // ---- references ----
    const ref_out ref = ref_lod(c, qh, kih, vih, keh, veh, kph, vph, scale);

    bool sel_ok = true;
    {
        const int64_t n_rows = c.shared ? Hkv : cols * Hkv;
        for (int64_t r = 0; r < n_rows && sel_ok; r++) {
            std::set<int64_t> got;
            for (int64_t i = 0; i < c.kP; i++) got.insert(sel_h[i + c.kP*r]);
            const int64_t kv  = c.shared ? r : r / cols;
            const int64_t col = c.shared ? 0 : r % cols;
            sel_ok = got == ref.sel[c.shared ? kv : kv * cols + col];
        }
    }

    const double e_ref = nmse(y.data(), ref.out.data(), (int64_t) y.size());

    double e_dense = 0.0;
    if (c.kP == c.P) {
        const std::vector<double> dense = ref_dense(c, qh, kih, vih, keh, veh, scale);
        e_dense = nmse(y.data(), dense.data(), (int64_t) y.size());
    }

    const bool ok = sel_ok && e_ref <= 1e-8 && e_dense <= 1e-8;
    printf("  lod %s D=%3d Hkv=%d g=%d ps=%2d P=%2d tail=%2d nq=%2d kP=%2d: sel=%s nmse_ref=%.2e",
           c.shared ? "shared" : "decode", (int) c.D, (int) c.Hkv, (int) c.g, (int) c.ps, (int) c.P,
           (int) c.n_tail, (int) c.nq, (int) c.kP, sel_ok ? "ok" : "MISMATCH", e_ref);
    if (c.kP == c.P) {
        printf(" nmse_dense=%.2e", e_dense);
    }
    printf(" %s\n", ok ? "OK" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(void) {
    int fails = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (backend == NULL) {
            printf("backend %s: init failed\n", ggml_backend_dev_name(dev));
            fails++;
            continue;
        }
        printf("backend %s:\n", ggml_backend_dev_name(dev));
        int unsupported = 0;
        for (const lod_case & c : CASES) {
            const int r = run_case(c, backend);
            if (r > 0) fails++;
            if (r < 0) unsupported++;
        }
        if (unsupported > 0) {
            printf("  (%d cases skipped for missing op support)\n", unsupported);
        }
        ggml_backend_free(backend);
    }
    printf(fails == 0 ? "all lod-attention checks passed\n" : "%d lod-attention checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
