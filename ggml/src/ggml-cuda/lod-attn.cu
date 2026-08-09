// LoD sparse attention read: one online softmax over three tiers
//   [unselected page summaries | leaves of the selected pages | raw tail + current tokens]
// split-K over a virtual column space, then a small merge kernel folds the partials.

#include "lod-attn.cuh"

#define LOD_ATTN_SPLITS   32
#if defined(GGML_USE_HIP)
#define LOD_ATTN_TILE     64
#define LOD_ATTN_NTILES    4
#else
#define LOD_ATTN_TILE     32
#define LOD_ATTN_NTILES    4
#endif
#define LOD_ATTN_MAX_D   512
#define LOD_ATTN_MAX_P  8192
#define LOD_ATTN_EPL2   (LOD_ATTN_MAX_D/(2*LOD_ATTN_TILE)) // max element pairs per lane
#define HPT                2  // query heads per tile (paired within one GQA group)

// one 32-lane tile processes one virtual column at a time; every lane owns two
// consecutive elements (D is even), enabling half2/float2 loads
static __global__ void lod_attn_split_f(
        const char * __restrict__ q,  const char * __restrict__ k,  const char * __restrict__ v,
        const char * __restrict__ ks, const char * __restrict__ vs,
        const int  * __restrict__ sel, const int * __restrict__ st,
        float      * __restrict__ part, // [n_rows, S, 2 + Dv]
        const int Dk, const int Dv, const int Hq, const int Hkv,
        const int P, const int n_sel, const int ps,
        const uint64_t q_nb1,  const uint64_t q_nb2,
        const uint64_t k_nb1,  const uint64_t k_nb2,
        const uint64_t v_nb1,  const uint64_t v_nb2,
        const uint64_t ks_nb1, const uint64_t ks_nb2,
        const uint64_t vs_nb1, const uint64_t vs_nb2,
        const float scale, const float logn, const int k_f16, const int v_f16) {
    const int row = blockIdx.x; // qi*(Hq/HPT) + hpair
    const int spl = blockIdx.y;

    const int nhp = Hq/HPT;
    const int qi  = row/nhp;
    const int h0  = (row%nhp)*HPT;
    const int hk  = h0/(Hq/Hkv);

    const int lane = threadIdx.x%LOD_ATTN_TILE;
    const int tile = threadIdx.x/LOD_ATTN_TILE;

    const int prev_end = st[0];
    const int Pr       = st[1];                  // complete pages to read
    const int limit    = prev_end + qi;          // last visible position, inclusive
    const int tail0    = Pr*ps;
    const int n_tail   = limit - tail0 + 1;
    const int n_cols   = Pr + n_sel*ps + n_tail; // virtual column space

    const int epl2_k = (Dk/2 + LOD_ATTN_TILE - 1)/LOD_ATTN_TILE; // element pairs per lane
    const int epl2_v = (Dv/2 + LOD_ATTN_TILE - 1)/LOD_ATTN_TILE;

    __shared__ float    sm_q[HPT][LOD_ATTN_MAX_D];
    __shared__ uint32_t sm_sel[LOD_ATTN_MAX_P/32];

    for (int j = 0; j < HPT; ++j) {
        for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
            sm_q[j][d] = ((const float *)(q + qi*q_nb1 + (uint64_t)(h0 + j)*q_nb2))[d];
        }
    }
    for (int i = threadIdx.x; i < (P + 31)/32; i += blockDim.x) {
        sm_sel[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        if (sel[i] >= 0) { // in-op selection pads with -1 when fewer pages exist
            atomicOr(&sm_sel[sel[i]/32], 1u << (sel[i]%32));
        }
    }
    __syncthreads();

    float m  [HPT];
    float den[HPT];
    float acc[HPT][LOD_ATTN_MAX_D/LOD_ATTN_TILE] = {{ 0.0f }}; // 2 consecutive elements per slot pair
    for (int j = 0; j < HPT; ++j) {
        m[j]   = -INFINITY;
        den[j] = 0.0f;
    }

    const int tile_g   = spl*LOD_ATTN_NTILES + tile;
    const int tile_stp = LOD_ATTN_SPLITS*LOD_ATTN_NTILES;

    for (int c = tile_g; c < n_cols; c += tile_stp) {
        const char * kp;
        const char * vp;

        float bias    = 0.0f;
        float vscale  = 1.0f;
        float kscale  = scale;
        int   kf16    = k_f16;
        int   vf16    = v_f16;

        if (c < Pr) {
            if (sm_sel[c/32] & (1u << (c%32))) {
                continue; // refinement: this page contributes its leaves instead
            }
            kp = ks + (uint64_t) c*ks_nb1 + hk*ks_nb2;
            vp = vs + (uint64_t) c*vs_nb1 + hk*vs_nb2;
            bias   = logn;
            vscale = 1.0f/ps;
            kscale = scale/ps;
            kf16   = 0;
            vf16   = 0;
        } else if (c < Pr + n_sel*ps) {
            const int pg = sel[(c - Pr)/ps];
            if (pg < 0) {
                continue; // in-op selection pads with -1 when fewer pages exist
            }
            const int col = pg*ps + (c - Pr)%ps;
            kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
            vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
        } else {
            const int col = tail0 + (c - Pr - n_sel*ps);
            kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
            vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
        }

        // dot(q, k): K loaded once per pair of heads, two consecutive elements per lane
        float sj[HPT] = { 0.0f };
        if (kf16) {
            const half2 * kh = (const half2 *) kp;
            for (int e = 0; e < epl2_k; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dk) {
                    const float2 kv2 = __half22float2(kh[i2]);
                    for (int j = 0; j < HPT; ++j) {
                        sj[j] += sm_q[j][2*i2]*kv2.x + sm_q[j][2*i2 + 1]*kv2.y;
                    }
                }
            }
        } else {
            const float2 * kf = (const float2 *) kp;
            for (int e = 0; e < epl2_k; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dk) {
                    const float2 kv2 = kf[i2];
                    for (int j = 0; j < HPT; ++j) {
                        sj[j] += sm_q[j][2*i2]*kv2.x + sm_q[j][2*i2 + 1]*kv2.y;
                    }
                }
            }
        }
#pragma unroll
        for (int off = LOD_ATTN_TILE/2; off > 0; off >>= 1) {
            for (int j = 0; j < HPT; ++j) {
                sj[j] += __shfl_xor_sync(0xffffffff, sj[j], off, LOD_ATTN_TILE);
            }
        }

        float w[HPT];
        for (int j = 0; j < HPT; ++j) {
            const float logit = sj[j]*kscale + bias;
            if (logit > m[j]) {
                const float e = expf(m[j] - logit);
                den[j] *= e;
                for (int i = 0; i < 2*epl2_v; ++i) {
                    acc[j][i] *= e;
                }
                m[j] = logit;
                w[j] = 1.0f;
            } else {
                w[j] = expf(logit - m[j]);
            }
            den[j] += w[j];
        }

        if (vf16) {
            const half2 * vh = (const half2 *) vp;
            for (int e = 0; e < epl2_v; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dv) {
                    const float2 vv2 = __half22float2(vh[i2]);
                    for (int j = 0; j < HPT; ++j) {
                        acc[j][2*e + 0] += w[j]*vscale*vv2.x;
                        acc[j][2*e + 1] += w[j]*vscale*vv2.y;
                    }
                }
            }
        } else {
            const float2 * vf = (const float2 *) vp;
            for (int e = 0; e < epl2_v; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dv) {
                    const float2 vv2 = vf[i2];
                    for (int j = 0; j < HPT; ++j) {
                        acc[j][2*e + 0] += w[j]*vscale*vv2.x;
                        acc[j][2*e + 1] += w[j]*vscale*vv2.y;
                    }
                }
            }
        }
    }

    // fold the tiles of this block through shared memory, one head at a time
    __shared__ float sm_m  [LOD_ATTN_NTILES];
    __shared__ float sm_den[LOD_ATTN_NTILES];
    __shared__ float sm_acc[LOD_ATTN_NTILES][LOD_ATTN_MAX_D];

    for (int j = 0; j < HPT; ++j) {
        __syncthreads();
        if (lane == 0) {
            sm_m[tile] = m[j];
        }
        __syncthreads();

        float m_b = -INFINITY;
        for (int t = 0; t < LOD_ATTN_NTILES; ++t) {
            m_b = fmaxf(m_b, sm_m[t]);
        }

        const float e_t = m[j] == -INFINITY ? 0.0f : expf(m[j] - m_b);

        if (lane == 0) {
            sm_den[tile] = den[j]*e_t;
        }
        for (int e = 0; e < epl2_v; ++e) {
            const int i2 = lane + e*LOD_ATTN_TILE;
            if (2*i2 < Dv) {
                sm_acc[tile][2*i2 + 0] = acc[j][2*e + 0]*e_t;
                sm_acc[tile][2*i2 + 1] = acc[j][2*e + 1]*e_t;
            }
        }
        __syncthreads();

        float * out = part + ((uint64_t)(qi*Hq + h0 + j)*LOD_ATTN_SPLITS + spl)*(2 + Dv);

        if (threadIdx.x == 0) {
            float den_b = 0.0f;
            for (int t = 0; t < LOD_ATTN_NTILES; ++t) {
                den_b += sm_den[t];
            }
            out[0] = m_b;
            out[1] = den_b;
        }
        for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
            float a = 0.0f;
            for (int t = 0; t < LOD_ATTN_NTILES; ++t) {
                a += sm_acc[t][d];
            }
            out[2 + d] = a;
        }
    }
}

// fold the current tokens into the page sums; their pages are >= P_read so the
// split kernel never reads them
static __global__ void lod_attn_fold_sums_f(
        const char * __restrict__ k, const char * __restrict__ v,
        float * __restrict__ ks, float * __restrict__ vs,
        const int * __restrict__ st,
        const int Dk, const int Dv, const int nq, const int ps,
        const uint64_t k_nb1, const uint64_t k_nb2, const uint64_t v_nb1, const uint64_t v_nb2,
        const uint64_t ks_nb1, const uint64_t ks_nb2, const uint64_t vs_nb1, const uint64_t vs_nb2,
        const int k_f16, const int v_f16) {
    const int hh = blockIdx.x;

    const int prev_end = st[0];

    for (int t = 0; t < nq; ++t) {
        const int col  = prev_end + t;
        const int page = col/ps;

        float       * ksd = (float *)((char *) ks + (uint64_t) page*ks_nb1 + hh*ks_nb2);
        const char  * kc  = k + (uint64_t) col*k_nb1 + hh*k_nb2;

        for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
            ksd[d] += k_f16 ? __half2float(((const half *) kc)[d]) : ((const float *) kc)[d];
        }

        float       * vsd = (float *)((char *) vs + (uint64_t) page*vs_nb1 + hh*vs_nb2);
        const char  * vc  = v + (uint64_t) col*v_nb1 + hh*v_nb2;

        for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
            vsd[d] += v_f16 ? __half2float(((const half *) vc)[d]) : ((const float *) vc)[d];
        }
    }
}

// in-op selection, stage 1: pooled page scores, plus the sums fold in the same launch.
// blocks < P*Hkv score one (page, KV head) each (max over queries and group heads of
// dot(q, k_sums); the ranking is scale-invariant so the raw sums are used); the last
// Hkv blocks run the fold of lod_attn_fold_sums_f (fold writes pages >= Pr, scoring
// reads pages < Pr - disjoint)
static __global__ void lod_attn_score_fold_f(
        const char * __restrict__ q, const char * __restrict__ ks,
        const char * __restrict__ k, const char * __restrict__ v,
        float * __restrict__ ksm, float * __restrict__ vsm,
        const int * __restrict__ st,
        float * __restrict__ scores,
        const int Dk, const int Dv, const int nq, const int Hq, const int Hkv,
        const int P, const int ps,
        const uint64_t q_nb1, const uint64_t q_nb2,
        const uint64_t ks_nb1, const uint64_t ks_nb2,
        const uint64_t vs_nb1, const uint64_t vs_nb2,
        const uint64_t k_nb1, const uint64_t k_nb2, const uint64_t v_nb1, const uint64_t v_nb2,
        const int k_f16, const int v_f16) {
    if ((int) blockIdx.x >= P*Hkv) {
        // fold path: one block per KV head
        const int hh       = blockIdx.x - P*Hkv;
        const int prev_end = st[0];

        for (int t = 0; t < nq; ++t) {
            const int col  = prev_end + t;
            const int page = col/ps;

            float      * ksd = (float *)((char *) ksm + (uint64_t) page*ks_nb1 + hh*ks_nb2);
            const char * kc  = k + (uint64_t) col*k_nb1 + hh*k_nb2;

            for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
                ksd[d] += k_f16 ? __half2float(((const half *) kc)[d]) : ((const float *) kc)[d];
            }

            float      * vsd = (float *)((char *) vsm + (uint64_t) page*vs_nb1 + hh*vs_nb2);
            const char * vc  = v + (uint64_t) col*v_nb1 + hh*v_nb2;

            for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
                vsd[d] += v_f16 ? __half2float(((const half *) vc)[d]) : ((const float *) vc)[d];
            }
        }
        return;
    }

    // one block (a single wave) per (page, KV head): lanes split the head dim so the
    // k_sums and q reads coalesce, one butterfly reduce per (head, query) pair;
    // scores land in a [Hkv, P] table and the top-k stage folds the KV rows
    const int bp = blockIdx.x;
    const int p  = bp/Hkv;
    const int kv = bp%Hkv;
    const int Pr = st[1];
    if (p >= Pr) {
        return; // incomplete page, never scored (topk only scans p < Pr)
    }

    const int g    = Hq/Hkv;
    const int lane = threadIdx.x;

    const float * ksp = (const float *)(ks + (uint64_t) p*ks_nb1 + (uint64_t) kv*ks_nb2);

    float ksd[2*LOD_ATTN_EPL2];
    for (int e = 0; e < Dk/LOD_ATTN_TILE; ++e) {
        ksd[e] = ksp[lane + e*LOD_ATTN_TILE];
    }

    float best = -INFINITY;
    for (int j = 0; j < g; ++j) {
        for (int qi = 0; qi < nq; ++qi) {
            const float * qp = (const float *)(q + (uint64_t) qi*q_nb1 + (uint64_t)(kv*g + j)*q_nb2);

            float s = 0.0f;
            for (int e = 0; e < Dk/LOD_ATTN_TILE; ++e) {
                s += qp[lane + e*LOD_ATTN_TILE]*ksd[e];
            }
#pragma unroll
            for (int off = LOD_ATTN_TILE/2; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffff, s, off, LOD_ATTN_TILE);
            }
            best = fmaxf(best, s);
        }
    }

    if (lane == 0) {
        scores[(uint64_t) kv*P + p] = best;
    }
}

// in-op selection, stage 2: pick the n_top best pages (ties -> lower page id,
// matching the CPU reference); -1 padded when fewer complete pages exist

// one shared-memory bitonic sort when the page capacity fits (<= 128k ctx at ps 64)
#define LOD_TOPK_SORT_MAX 2048

static __global__ void lod_attn_topk_sort(
        const float * __restrict__ scores, int * __restrict__ sel,
        const int * __restrict__ st, const int n_top, const int P, const int Hkv) {
    const int Pr = st[1];

    __shared__ float sv[LOD_TOPK_SORT_MAX];
    __shared__ int   si[LOD_TOPK_SORT_MAX];

    int n = 1;
    while (n < Pr) {
        n <<= 1;
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = -INFINITY;
        if (i < Pr) {
            for (int kv = 0; kv < Hkv; ++kv) {
                v = fmaxf(v, scores[(uint64_t) kv*P + i]);
            }
        }
        sv[i] = v;
        si[i] = i;
    }
    __syncthreads();

    for (int ksz = 2; ksz <= n; ksz <<= 1) {
        for (int j = ksz >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    // descending by score, ascending by page id on ties
                    const bool keep = sv[i] > sv[ixj] || (sv[i] == sv[ixj] && si[i] < si[ixj]);
                    if (((i & ksz) == 0) ? !keep : keep) {
                        const float tv = sv[i]; sv[i] = sv[ixj]; sv[ixj] = tv;
                        const int   ti = si[i]; si[i] = si[ixj]; si[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < n_top; i += blockDim.x) {
        sel[i] = i < Pr ? si[i] : -1;
    }
}

// fallback above the sort capacity: iterative extraction
static __global__ void lod_attn_topk(
        float * __restrict__ scores, int * __restrict__ sel,
        const int * __restrict__ st, const int n_top, const int P, const int Hkv) {
    const int Pr = st[1];

    __shared__ float sm_v[256];
    __shared__ int   sm_i[256];

    for (int i = threadIdx.x; i < Pr; i += blockDim.x) {
        float v = scores[i];
        for (int kv = 1; kv < Hkv; ++kv) {
            v = fmaxf(v, scores[(uint64_t) kv*P + i]);
        }
        scores[i] = v;
    }
    __syncthreads();

    for (int it = 0; it < n_top; ++it) {
        float bv = -INFINITY;
        int   bi = -1;
        for (int p = threadIdx.x; p < Pr; p += blockDim.x) {
            if (scores[p] > bv) {
                bv = scores[p];
                bi = p;
            }
        }
        sm_v[threadIdx.x] = bv;
        sm_i[threadIdx.x] = bi;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if ((int) threadIdx.x < off) {
                const float ov = sm_v[threadIdx.x + off];
                const int   oi = sm_i[threadIdx.x + off];
                if (oi >= 0 && (ov > sm_v[threadIdx.x] || (ov == sm_v[threadIdx.x] && (sm_i[threadIdx.x] < 0 || oi < sm_i[threadIdx.x])))) {
                    sm_v[threadIdx.x] = ov;
                    sm_i[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            sel[it] = sm_i[0];
            if (sm_i[0] >= 0) {
                scores[sm_i[0]] = -INFINITY;
            }
        }
        __syncthreads();
    }
}

// grid (n_rows, Dv/MERGE_SLICE): every block redundantly folds the tiny split
// headers, then combines its own slice of the value dimension - enough blocks
// to cover the whole GPU instead of one block per row
#define LOD_ATTN_MERGE_SLICE 64

static __global__ void lod_attn_merge_f(
        const float * __restrict__ part, float * __restrict__ dst,
        const int Dv, const int Hq) {
    const int row = blockIdx.x; // qi*Hq + h
    const int d0  = blockIdx.y*LOD_ATTN_MERGE_SLICE;

    const float * p = part + (uint64_t) row*LOD_ATTN_SPLITS*(2 + Dv);

    __shared__ float sm_w[LOD_ATTN_SPLITS];

    float m = -INFINITY;
    for (int s = 0; s < LOD_ATTN_SPLITS; ++s) {
        m = fmaxf(m, p[s*(2 + Dv)]);
    }

    for (int s = threadIdx.x; s < LOD_ATTN_SPLITS; s += blockDim.x) {
        const float ms = p[s*(2 + Dv)];
        sm_w[s] = ms > -INFINITY/2 ? expf(ms - m) : 0.0f;
    }
    __syncthreads();

    // block-wide denominator: every thread re-sums the shared weights
    float den = 0.0f;
    for (int s = 0; s < LOD_ATTN_SPLITS; ++s) {
        den += p[s*(2 + Dv) + 1]*sm_w[s];
    }

    const float id = den > 0.0f ? 1.0f/den : 0.0f;

    // dst layout [Dv, Hq, nq]: row = qi*Hq + h maps directly
    float * out = dst + (uint64_t) row*Dv;

    for (int d = d0 + threadIdx.x; d < min(d0 + LOD_ATTN_MERGE_SLICE, Dv); d += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < LOD_ATTN_SPLITS; ++s) {
            const float w = sm_w[s];
            if (w > 0.0f) {
                a += p[s*(2 + Dv) + 2 + d]*w;
            }
        }
        out[d] = a*id;
    }
}

bool ggml_cuda_lod_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * k  = dst->src[1];
    const ggml_tensor * v  = dst->src[2];
    const ggml_tensor * ks = dst->src[3];

    return q->ne[0] <= LOD_ATTN_MAX_D && v->ne[0] <= LOD_ATTN_MAX_D &&
           q->ne[0]%2 == 0 && v->ne[0]%2 == 0 &&
           (q->ne[2]/k->ne[2])%HPT == 0 &&
           q->ne[2]/k->ne[2] <= 32 && // in-op selection: bounded GQA group
           q->ne[0]%LOD_ATTN_TILE == 0 && // in-op selection: lanes split the head dim evenly

           ks->ne[1] <= LOD_ATTN_MAX_P &&
           (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) &&
           (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32) &&
           q->nb[0] == sizeof(float);
}

void ggml_cuda_lod_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * ks  = dst->src[3];
    const ggml_tensor * vs  = dst->src[4];
    const ggml_tensor * sel = dst->src[5];
    const ggml_tensor * st  = dst->src[6];

    const int32_t ps = ggml_get_op_params_i32(dst, 0);

    float scale;
    memcpy(&scale, (const char *) dst->op_params + sizeof(int32_t), sizeof(float));

    const int Dk  = q->ne[0];
    const int nq  = q->ne[1];
    const int Hq  = q->ne[2];
    const int Dv  = v->ne[0];
    const int Hkv = k->ne[2];
    const int P   = ks->ne[1];

    const int n_rows = nq*Hq;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) n_rows*LOD_ATTN_SPLITS*(2 + Dv));

    const int k_f16 = k->type == GGML_TYPE_F16;
    const int v_f16 = v->type == GGML_TYPE_F16;

    const int32_t * sel_ptr;
    int             n_sel;

    ggml_cuda_pool_alloc<float>   ws_scores(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ws_sel(ctx.pool());

    if (sel != nullptr) {
        sel_ptr = (const int32_t *) sel->data;
        n_sel   = (int) sel->ne[0];

        lod_attn_fold_sums_f<<<Hkv, 256, 0, stream>>>(
                (const char *) k->data, (const char *) v->data,
                (float *) ks->data, (float *) vs->data, (const int *) st->data,
                Dk, Dv, nq, ps,
                k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
                k_f16, v_f16);
    } else {
        // in-op selection: pooled page scores + fold in one launch, then a tiny top-k
        n_sel = ggml_get_op_params_i32(dst, 2);

        ws_scores.alloc((size_t) P*Hkv);
        ws_sel.alloc(n_sel);

        lod_attn_score_fold_f<<<P*Hkv + Hkv, LOD_ATTN_TILE, 0, stream>>>(
                (const char *) q->data, (const char *) ks->data,
                (const char *) k->data, (const char *) v->data,
                (float *) ks->data, (float *) vs->data, (const int *) st->data,
                ws_scores.get(),
                Dk, Dv, nq, Hq, Hkv, P, ps,
                q->nb[1], q->nb[2],
                ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
                k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                k_f16, v_f16);

        if (P <= LOD_TOPK_SORT_MAX) {
            lod_attn_topk_sort<<<1, 256, 0, stream>>>(ws_scores.get(), ws_sel.get(), (const int *) st->data, n_sel, P, Hkv);
        } else {
            lod_attn_topk<<<1, 256, 0, stream>>>(ws_scores.get(), ws_sel.get(), (const int *) st->data, n_sel, P, Hkv);
        }

        sel_ptr = ws_sel.get();
    }

    const dim3 grid(n_rows/HPT, LOD_ATTN_SPLITS, 1);
    const dim3 block(LOD_ATTN_TILE*LOD_ATTN_NTILES, 1, 1);

    lod_attn_split_f<<<grid, block, 0, stream>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const char *) ks->data, (const char *) vs->data,
            sel_ptr, (const int *) st->data,
            part.get(),
            Dk, Dv, Hq, Hkv, P, n_sel, ps,
            q->nb[1],  q->nb[2],  k->nb[1],  k->nb[2],  v->nb[1], v->nb[2],
            ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
            scale, logf((float) ps), k_f16, v_f16);

    const dim3 grid_m(n_rows, (Dv + LOD_ATTN_MERGE_SLICE - 1)/LOD_ATTN_MERGE_SLICE, 1);

    lod_attn_merge_f<<<grid_m, LOD_ATTN_MERGE_SLICE, 0, stream>>>(part.get(), (float *) dst->data, Dv, Hq);
}
