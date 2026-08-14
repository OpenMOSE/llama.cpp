// LoD2 attention (ref. "Key-Value Means", arXiv:2605.09877) - CPU reference.
//
// This implementation is written for clarity, not for speed: it is the
// correctness oracle that the accelerated backends are checked against, and it
// follows docs/lod2-port-spec.md step by step.  Threads are handed whole
// (kv head) or (token, query head) units, which is enough parallelism for the
// tests without any cross-thread synchronization inside an op.

#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "traits.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// helpers

namespace {

// read one k/v column (a [D] row at column `col` of head `h`) as float
struct lod2_reader {
    const ggml_tensor * t;
    ggml_to_float_t     to_float;
    int64_t             n;

    lod2_reader(const ggml_tensor * t) : t(t), n(t->ne[0]) {
        to_float = t->type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(t->type)->to_float;
        GGML_ASSERT(t->type == GGML_TYPE_F32 || to_float != nullptr);
    }

    const float * row(int64_t col, int64_t h, float * scratch) const {
        const char * src = (const char *) t->data + col*t->nb[1] + h*t->nb[2];
        if (to_float == nullptr) {
            return (const float *) src;
        }
        to_float(src, scratch, n);
        return scratch;
    }
};

inline float lod2_dot(const float * a, const float * b, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        s += a[i]*b[i];
    }
    return s;
}

// online softmax accumulator: feed logits one at a time, get (output, lse)
struct lod2_branch {
    float          m = -INFINITY; // running max
    float          d = 0.0f;      // running sum of exp(logit - m)
    std::vector<float> acc;

    void reset(int64_t dv) {
        m = -INFINITY;
        d = 0.0f;
        acc.assign(dv, 0.0f);
    }

    void add(float logit, const float * value) {
        if (!(logit > -INFINITY)) {
            return;
        }
        const int64_t dv = (int64_t) acc.size();
        if (logit > m) {
            const float c = m == -INFINITY ? 0.0f : expf(m - logit);
            for (int64_t i = 0; i < dv; ++i) {
                acc[i] *= c;
            }
            d = d*c + 1.0f;
            m = logit;
            for (int64_t i = 0; i < dv; ++i) {
                acc[i] += value[i];
            }
            return;
        }
        const float p = expf(logit - m);
        d += p;
        for (int64_t i = 0; i < dv; ++i) {
            acc[i] += p*value[i];
        }
    }

    float lse() const {
        return d > 0.0f ? m + logf(d) : -INFINITY;
    }

    // normalized output; undefined (and unused) when lse() is -inf
    void normalized(float * out) const {
        const int64_t dv = (int64_t) acc.size();
        const float inv = d > 0.0f ? 1.0f/d : 0.0f;
        for (int64_t i = 0; i < dv; ++i) {
            out[i] = acc[i]*inv;
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// ggml_compute_forward_lod2_update

void ggml_compute_forward_lod2_update(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * k     = dst->src[0];
    const ggml_tensor * v     = dst->src[1];
    const ggml_tensor * s_kv  = dst->src[2];
    const ggml_tensor * p_kv  = dst->src[3];
    const ggml_tensor * p_idx = dst->src[4];
    const ggml_tensor * s_pg  = dst->src[5];
    const ggml_tensor * meta  = dst->src[6];
    const ggml_tensor * s_mn  = dst->src[7];

    const int32_t p0        = ggml_get_op_params_i32(dst, 0);
    const int32_t p1        = ggml_get_op_params_i32(dst, 1);
    const int32_t s_len     = ggml_get_op_params_i32(dst, 2);
    const int32_t s_len_new = ggml_get_op_params_i32(dst, 3);
    const int32_t sink_len  = ggml_get_op_params_i32(dst, 4);

    const int64_t D   = k->ne[0];
    const int64_t Dv  = v->ne[0];
    const int64_t Ds  = s_kv->ne[0];
    const int64_t Hkv = k->ne[2];
    const int64_t ps  = p_idx->ne[0];
    const int64_t pps = s_pg->ne[0] - 1;

    const int64_t M         = p1 - p0;
    const int64_t n_append  = s_len_new - s_len;
    const int64_t P_cap     = p_kv->ne[1];

    // padding block: the graph carries a fixed number of these so its node
    // count is stable, and an empty one has nothing to absorb
    if (M == 0) {
        return;
    }

    GGML_ASSERT(Ds == D + Dv + 1);

    const lod2_reader kr(k);
    const lod2_reader vr(v);

    std::vector<float> ks(D), vs(Dv), mean(D);
    std::vector<float> block_k((size_t) M*D), block_v((size_t) M*Dv);
    std::vector<float> best_score(M), select(M);
    std::vector<int32_t> best_slot(M), owner(M, -1), order(M);

    for (int64_t h = params->ith; h < Hkv; h += params->nth) {
        float   * s_data = (float   *)((char *) s_kv->data);
        float   * p_data = (float   *)((char *) p_kv->data);
        int32_t * i_data = (int32_t *)((char *) p_idx->data);
        int32_t * g_data = (int32_t *)((char *) s_pg->data);
        int32_t * m_data = (int32_t *)((char *) meta->data);

        float   * slot  = s_data + h*s_kv->ne[1]*Ds;
        float   * page  = p_data + h*P_cap*Ds;
        int32_t * pidx  = i_data + h*P_cap*ps;
        int32_t * spg   = g_data + h*s_pg->ne[1]*(pps + 1);
        int32_t * hmeta = m_data + h*meta->ne[0];

        // cache the block's K/V once: every column is read many times below
        for (int64_t t = 0; t < M; ++t) {
            memcpy(&block_k[(size_t) t*D],  kr.row(p0 + t, h, ks.data()), D *sizeof(float));
            memcpy(&block_v[(size_t) t*Dv], vr.row(p0 + t, h, vs.data()), Dv*sizeof(float));
        }

        if (s_len == 0) {
            // every column of the very first block becomes its own slot
            GGML_ASSERT(n_append == M);
            for (int64_t t = 0; t < M; ++t) {
                float * row = slot + t*Ds;
                memcpy(row,     &block_k[(size_t) t*D],  D *sizeof(float));
                memcpy(row + D, &block_v[(size_t) t*Dv], Dv*sizeof(float));
                row[D + Dv] = 1.0f;
                owner[t] = (int32_t) t;
            }
        } else {
            GGML_ASSERT(s_len > sink_len);

            // similarity of each new column against the live slot means
            for (int64_t t = 0; t < M; ++t) {
                const float * kt = &block_k[(size_t) t*D];
                float best = -INFINITY;
                float prot = -INFINITY;
                int32_t arg = 0;
                for (int64_t s = 0; s < s_len; ++s) {
                    const float * row = slot + s*Ds;
                    const float cnt = row[D + Dv];
                    if (!(cnt > 0.5f)) {
                        continue;
                    }
                    const float inv = 1.0f/cnt;
                    for (int64_t i = 0; i < D; ++i) {
                        mean[i] = row[i]*inv;
                    }
                    const float sim = lod2_dot(kt, mean.data(), D);
                    if (s < sink_len) {
                        // the sink is never a merge target, but a column that
                        // matches it is not "novel" either
                        prot = std::max(prot, sim);
                    } else if (sim > best) {
                        best = sim;
                        arg  = (int32_t) s;
                    }
                }
                best_score[t] = best;
                best_slot[t]  = arg;
                select[t]     = std::max(best, prot);
            }

            // the least familiar columns become new slots
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                [&](int32_t a, int32_t b) { return select[a] < select[b]; });
            std::sort(order.begin(), order.begin() + n_append);
            std::sort(order.begin() + n_append, order.end());

            for (int64_t j = 0; j < n_append; ++j) {
                const int64_t t   = order[j];
                float       * row = slot + (s_len + j)*Ds;
                memcpy(row,     &block_k[(size_t) t*D],  D *sizeof(float));
                memcpy(row + D, &block_v[(size_t) t*Dv], Dv*sizeof(float));
                row[D + Dv] = 1.0f;
                owner[t] = (int32_t) (s_len + j);
            }

            // the rest merge into the best of (old non-sink slots, new slots),
            // all judged against the state as it was before this block
            for (int64_t j = n_append; j < M; ++j) {
                const int64_t t  = order[j];
                const float * kt = &block_k[(size_t) t*D];
                float   best = best_score[t];
                int32_t dest = best_slot[t];
                for (int64_t a = 0; a < n_append; ++a) {
                    const float sim = lod2_dot(kt, &block_k[(size_t) order[a]*D], D);
                    if (sim > best) {
                        best = sim;
                        dest = (int32_t) (s_len + a);
                    }
                }
                owner[t] = dest;
            }
            for (int64_t j = n_append; j < M; ++j) {
                const int64_t t   = order[j];
                float       * row = slot + owner[t]*Ds;
                for (int64_t i = 0; i < D; ++i) {
                    row[i] += block_k[(size_t) t*D + i];
                }
                for (int64_t i = 0; i < Dv; ++i) {
                    row[D + i] += block_v[(size_t) t*Dv + i];
                }
                row[D + Dv] += 1.0f;
            }
        }

        // page append: each slot's tokens are paged in arrival order
        for (int64_t t = 0; t < M; ++t) {
            const int32_t s   = owner[t];
            int32_t     * ent = spg + s*(pps + 1);
            const int32_t len = ent[pps];
            const int64_t ord = len/ps;
            const int64_t ln  = len%ps;
            GGML_ASSERT(ord < pps);

            int32_t p;
            if (ln == 0) {
                p = hmeta[0]++;
                GGML_ASSERT(p < P_cap);
                ent[ord] = p;
            } else {
                p = ent[ord];
            }

            pidx[p*ps + ln] = (int32_t) (p0 + t);

            float * prow = page + p*Ds;
            for (int64_t i = 0; i < D; ++i) {
                prow[i] += block_k[(size_t) t*D + i];
            }
            for (int64_t i = 0; i < Dv; ++i) {
                prow[D + i] += block_v[(size_t) t*Dv + i];
            }
            prow[D + Dv] += 1.0f;

            ent[pps] = len + 1;
        }

        // publish the means the read path uses
        float * mn = (float *)((char *) s_mn->data) + h*s_kv->ne[1]*(Ds - 1);
        for (int64_t s = 0; s < s_len_new; ++s) {
            const float * row = slot + s*Ds;
            const float cnt = row[D + Dv];
            const float inv = cnt > 0.0f ? 1.0f/cnt : 0.0f;
            for (int64_t i = 0; i < Ds - 1; ++i) {
                mn[s*(Ds - 1) + i] = row[i]*inv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_lod2_attn

void ggml_compute_forward_lod2_attn(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * s_kv  = dst->src[3];
    const ggml_tensor * p_kv  = dst->src[4];
    const ggml_tensor * p_idx = dst->src[5];
    const ggml_tensor * s_pg  = dst->src[6];
    const ggml_tensor * meta  = dst->src[7];
    const ggml_tensor * s_mn  = dst->src[8];
    const ggml_tensor * lgt   = dst->src[9];

    const int32_t l0       = ggml_get_op_params_i32(dst, 0);
    const int32_t s_len    = ggml_get_op_params_i32(dst, 1);
    const int32_t n_routes = ggml_get_op_params_i32(dst, 2);
    const int32_t sink_len = ggml_get_op_params_i32(dst, 3);

    const int32_t q0 = ((const int32_t *) meta->data)[0];

    float scale;
    memcpy(&scale, (const char *) dst->op_params + 4*sizeof(int32_t), sizeof(float));

    const int64_t D   = q->ne[0];
    const int64_t nq  = q->ne[1];
    const int64_t Hq  = q->ne[2];
    const int64_t Dv  = v->ne[0];
    const int64_t Hkv = k->ne[2];
    const int64_t Ds  = s_kv->ne[0];
    const int64_t ps  = p_idx->ne[0];
    const int64_t pps = s_pg->ne[0] - 1;
    const int64_t g   = Hq/Hkv;

    const int64_t S_cap = s_kv->ne[1];
    const int64_t P_cap = p_kv->ne[1];

    const int32_t protect = (int32_t) std::min<int64_t>(sink_len, s_len);
    const int32_t R       = (int32_t) std::min<int64_t>(n_routes, std::max<int64_t>(s_len - protect, 0));

    const lod2_reader kr(k);
    const lod2_reader vr(v);

    const float * s_data = (const float   *) s_kv->data;
    const float * m_data = (const float *) s_mn->data;
    const float * p_data = (const float   *) p_kv->data;
    const int32_t * i_data = (const int32_t *) p_idx->data;
    const int32_t * g_data = (const int32_t *) s_pg->data;

    std::vector<float>   krow(D), vrow(Dv), mean(std::max(D, Dv)), work(std::max(D, Dv));
    std::vector<float>   logit(S_cap);
    std::vector<int32_t> routes(std::max(R, 1));
    std::vector<char>    opened(S_cap);
    lod2_branch coarse, local, exact;
    std::vector<float> out_coarse(Dv), out_local(Dv), out_exact(Dv);

    const int64_t n_work = nq*Hq;

    for (int64_t w = params->ith; w < n_work; w += params->nth) {
        const int64_t t  = w/Hq;
        const int64_t hq = w%Hq;
        const int64_t h  = hq/g;

        const float * qv  = (const float *)((const char *) q->data + t*q->nb[1] + hq*q->nb[2]);
        // the GEMM is optional (see ggml_lod2_attn): with one query it costs
        // more than the contraction is worth, so it may not have been run
        const float * lgv = lgt ? (const float *) lgt->data + (hq*nq + t)*lgt->ne[0] : nullptr;
        const float * slot = s_data + h*S_cap*Ds;
        const float * smean = m_data + h*S_cap*(Ds - 1);
        const float * page = p_data + h*P_cap*Ds;
        const int32_t * pidx = i_data + h*P_cap*ps;
        const int32_t * spg  = g_data + h*S_cap*(pps + 1);

        // --- 5.1 routing -----------------------------------------------------
        for (int64_t s = 0; s < s_len; ++s) {
            const float * row = slot + s*Ds;
            const float cnt = row[D + Dv];
            if (!(cnt > 0.5f)) {
                logit[s] = -INFINITY;
                continue;
            }
            float dot;
            if (lgv) {
                dot = lgv[s];
            } else {
                const float * mk = smean + s*(Ds - 1);
                dot = 0.0f;
                for (int64_t i = 0; i < D; ++i) {
                    dot += qv[i]*mk[i];
                }
            }
            logit[s] = scale*dot + logf(cnt);
        }

        std::fill(opened.begin(), opened.begin() + s_len, 0);
        int32_t n_open = 0;
        for (int32_t r = 0; r < R; ++r) {
            int64_t arg  = -1;
            float   best = -INFINITY;
            for (int64_t s = protect; s < s_len; ++s) {
                if (opened[s]) {
                    continue;
                }
                if (logit[s] > best) {
                    best = logit[s];
                    arg  = s;
                }
            }
            if (arg < 0 || !(best > -INFINITY)) {
                break; // no candidate left with any mass
            }
            opened[arg] = 1;
            routes[n_open++] = (int32_t) arg;
        }

        // --- 5.2 coarse branch ----------------------------------------------
        coarse.reset(Dv);
        for (int64_t s = 0; s < s_len; ++s) {
            if (opened[s] || !(logit[s] > -INFINITY)) {
                continue;
            }
            const float * mrow = smean + s*(Ds - 1);
            for (int64_t i = 0; i < Dv; ++i) {
                work[i] = mrow[D + i];
            }
            coarse.add(logit[s], work.data());
        }

        // --- 5.3 local branch ------------------------------------------------
        local.reset(Dv);
        for (int64_t j = l0; j <= q0 + t; ++j) {
            const float * kj = kr.row(j, h, krow.data());
            const float * vj = vr.row(j, h, vrow.data());
            local.add(scale*lod2_dot(qv, kj, D), vj);
        }

        // --- 5.4 routed branch -----------------------------------------------
        exact.reset(Dv);
        for (int32_t r = 0; r < n_open; ++r) {
            const int32_t s   = routes[r];
            const int32_t * ent = spg + s*(pps + 1);
            const int64_t n_pg = (ent[pps] + ps - 1)/ps;

            int32_t sel  = -1;
            float   best = -INFINITY;
            for (int64_t o = 0; o < n_pg; ++o) {
                const int32_t p = ent[o];
                if (p < 0) {
                    continue;
                }
                const float * prow = page + p*Ds;
                const float cnt = prow[D + Dv];
                if (!(cnt > 0.0f)) {
                    continue;
                }
                const float inv = 1.0f/cnt;
                for (int64_t i = 0; i < D; ++i) {
                    mean[i] = prow[i]*inv;
                }
                const float sc = scale*lod2_dot(qv, mean.data(), D) + logf(cnt);
                if (sc > best) {
                    best = sc;
                    sel  = p;
                }
            }
            if (sel < 0) {
                continue;
            }

            const float * prow = page + sel*Ds;
            const float * srow = slot + s*Ds;
            const float p_cnt  = prow[D + Dv];
            const float s_cnt  = srow[D + Dv];

            // one count-corrected residual standing for the rest of the slot
            const float r_cnt = s_cnt - p_cnt;
            if (r_cnt > 0.0f) {
                const float inv = 1.0f/r_cnt;
                for (int64_t i = 0; i < D; ++i) {
                    mean[i] = (srow[i] - prow[i])*inv;
                }
                const float sc = scale*lod2_dot(qv, mean.data(), D) + logf(r_cnt);
                for (int64_t i = 0; i < Dv; ++i) {
                    mean[i] = (srow[D + i] - prow[D + i])*inv;
                }
                exact.add(sc, mean.data());
            }

            // and the winning page read exactly
            const int64_t n_tok = std::min<int64_t>((int64_t) p_cnt, ps);
            for (int64_t i = 0; i < n_tok; ++i) {
                const int32_t col = pidx[sel*ps + i];
                if (col < 0) {
                    continue;
                }
                const float * kc = kr.row(col, h, krow.data());
                const float * vc = vr.row(col, h, vrow.data());
                exact.add(scale*lod2_dot(qv, kc, D), vc);
            }
        }

        // --- 5.5 merge --------------------------------------------------------
        const float lc = coarse.lse();
        const float ll = local.lse();
        const float le = exact.lse();
        const float m  = std::max(lc, std::max(ll, le));
        GGML_ASSERT(m > -INFINITY);

        const float wc = lc > -INFINITY ? expf(lc - m) : 0.0f;
        const float wl = ll > -INFINITY ? expf(ll - m) : 0.0f;
        const float we = le > -INFINITY ? expf(le - m) : 0.0f;
        const float inv = 1.0f/(wc + wl + we);

        coarse.normalized(out_coarse.data());
        local .normalized(out_local .data());
        exact .normalized(out_exact .data());

        float * out = (float *)((char *) dst->data + hq*dst->nb[1] + t*dst->nb[2]);
        for (int64_t i = 0; i < Dv; ++i) {
            out[i] = (wc*out_coarse[i] + wl*out_local[i] + we*out_exact[i])*inv;
        }
    }
}
