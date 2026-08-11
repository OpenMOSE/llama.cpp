// LoD sparse attention read: one online softmax over three tiers
//   [unselected page summaries | leaves of the selected pages | raw tail + current tokens]
// split-K over a virtual column space, then a small merge kernel folds the partials.

#include "lod-attn.cuh"

#include <type_traits>

#define LOD_ATTN_SPLITS      32 // fallback split-K width
#define LOD_ATTN_BPC          4 // split blocks targeted per compute unit
#define LOD_ATTN_SPLITS_LDS 128 // the LDS variant has Hkv x-blocks instead of Hq/HPT
#define LOD_ATTN_NTILES    4  // mega ablation only; the split kernel derives its own
#define LOD_ATTN_BLOCK   256  // lanes per split block (NTILES waves of the device width)
#define LOD_ATTN_MAX_D   512
#define LOD_ATTN_MAX_P  8192
#define HPT                2  // query heads per tile (paired within one GQA group)

// K/V type codes carried in the (historically named) k_f16/v_f16 params:
// 0 = F32, 1 = F16, 2 = Q8_0. two consecutive elements always share one
// 32-element q8_0 block, so the pairwise loads stay single-block
static __device__ __forceinline__ float2 lod_load2_q8(const char * p, const int i2) {
    const block_q8_0 * b = (const block_q8_0 *) p;
    const int ib = i2/(QK8_0/2);
    const int iq = 2*(i2%(QK8_0/2));
    const float d = __half2float(b[ib].d);
    return make_float2(d*b[ib].qs[iq + 0], d*b[ib].qs[iq + 1]);
}
static __device__ __forceinline__ float lod_load1_q8(const char * p, const int d) {
    const block_q8_0 * b = (const block_q8_0 *) p;
    return __half2float(b[d/QK8_0].d)*b[d/QK8_0].qs[d%QK8_0];
}

// the tile is one hardware wave: 64 on CDNA (wave64), 32 on RDNA and NVIDIA - shuffles
// wider than the wave return garbage, so the tile size is picked at run time from the
// device's warp size and both instantiations are compiled

// one wave-wide tile processes one virtual column at a time; every lane owns two
// consecutive elements (D is even), enabling half2/float2 loads.
// EPL2V (value element pairs per lane) is a template parameter so the accumulator is
// indexed by compile-time constants - with a run-time bound the whole array lands in
// scratch. The block spans LOD_ATTN_BLOCK lanes on every device, so a 32-wide wave
// (NVIDIA, RDNA) gets twice the tiles of a 64-wide one (CDNA) and both cover the same
// number of virtual columns per block
template <int tile_sz, int EPL2V>
static __global__ void lod_attn_split_f(
        const char * __restrict__ q,  const char * __restrict__ k,  const char * __restrict__ v,
        const char * __restrict__ ks, const char * __restrict__ vs,
        const int  * __restrict__ sel_in, const int * __restrict__ st,
        float      * __restrict__ part, // [n_rows, S, 2 + Dv]
        const int Dk, const int Dv, const int Hq, const int Hkv,
        const int P, const int n_sel, const int ps,
        const uint64_t q_nb1,  const uint64_t q_nb2,
        const uint64_t k_nb1,  const uint64_t k_nb2,
        const uint64_t v_nb1,  const uint64_t v_nb2,
        const uint64_t ks_nb1, const uint64_t ks_nb2,
        const uint64_t vs_nb1, const uint64_t vs_nb2,
        const float scale, const float logn, const int k_f16, const int v_f16,
        const int sel_head, const int nsplit) {
    const int row = blockIdx.x; // qi*(Hq/HPT) + hpair
    const int spl = blockIdx.y;

    const int nhp = Hq/HPT;
    const int qi  = row/nhp;
    const int h0  = (row%nhp)*HPT;
    const int hk  = h0/(Hq/Hkv);

    const int lane = threadIdx.x%tile_sz;
    const int tile = threadIdx.x/tile_sz;

    const int * sel = sel_in + (sel_head ? hk*n_sel : 0);

    const int prev_end = st[0];
    const int Pr       = st[1];                  // complete pages to read
    const int limit    = prev_end + qi;          // last visible position, inclusive
    const int tail0    = Pr*ps;
    const int n_tail   = limit - tail0 + 1;
    const int n_cols   = Pr + n_sel*ps + n_tail; // virtual column space

    const int epl2_k = (Dk/2 + tile_sz - 1)/tile_sz; // element pairs per lane

    constexpr int NT  = LOD_ATTN_BLOCK/tile_sz; // tiles (waves) per block
    constexpr int DVT = 2*EPL2V*tile_sz;        // value elements one tile covers

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
    float acc[HPT][2*EPL2V] = {{ 0.0f }}; // 2 consecutive elements per slot pair
    for (int j = 0; j < HPT; ++j) {
        m[j]   = -INFINITY;
        den[j] = 0.0f;
    }

    const int tile_g   = spl*NT + tile;
    const int tile_stp = nsplit*NT;

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
        if (kf16 == 2) {
            for (int e = 0; e < epl2_k; ++e) {
                const int i2 = lane + e*tile_sz;
                if (2*i2 < Dk) {
                    const float2 kv2 = lod_load2_q8(kp, i2);
                    for (int j = 0; j < HPT; ++j) {
                        sj[j] += sm_q[j][2*i2]*kv2.x + sm_q[j][2*i2 + 1]*kv2.y;
                    }
                }
            }
        } else if (kf16) {
            const half2 * kh = (const half2 *) kp;
            for (int e = 0; e < epl2_k; ++e) {
                const int i2 = lane + e*tile_sz;
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
                const int i2 = lane + e*tile_sz;
                if (2*i2 < Dk) {
                    const float2 kv2 = kf[i2];
                    for (int j = 0; j < HPT; ++j) {
                        sj[j] += sm_q[j][2*i2]*kv2.x + sm_q[j][2*i2 + 1]*kv2.y;
                    }
                }
            }
        }
#pragma unroll
        for (int off = tile_sz/2; off > 0; off >>= 1) {
            for (int j = 0; j < HPT; ++j) {
                sj[j] += __shfl_xor_sync(0xffffffff, sj[j], off, tile_sz);
            }
        }

        float w[HPT];
        for (int j = 0; j < HPT; ++j) {
            const float logit = sj[j]*kscale + bias;
            if (logit > m[j]) {
                const float e = expf(m[j] - logit);
                den[j] *= e;
#pragma unroll
                for (int i = 0; i < 2*EPL2V; ++i) {
                    acc[j][i] *= e;
                }
                m[j] = logit;
                w[j] = 1.0f;
            } else {
                w[j] = expf(logit - m[j]);
            }
            den[j] += w[j];
        }

        if (vf16 == 2) {
#pragma unroll
            for (int e = 0; e < EPL2V; ++e) {
                const int i2 = lane + e*tile_sz;
                if (2*i2 < Dv) {
                    const float2 vv2 = lod_load2_q8(vp, i2);
                    for (int j = 0; j < HPT; ++j) {
                        acc[j][2*e + 0] += w[j]*vscale*vv2.x;
                        acc[j][2*e + 1] += w[j]*vscale*vv2.y;
                    }
                }
            }
        } else if (vf16) {
            const half2 * vh = (const half2 *) vp;
#pragma unroll
            for (int e = 0; e < EPL2V; ++e) {
                const int i2 = lane + e*tile_sz;
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
#pragma unroll
            for (int e = 0; e < EPL2V; ++e) {
                const int i2 = lane + e*tile_sz;
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
    __shared__ float sm_m  [NT];
    __shared__ float sm_den[NT];
    __shared__ float sm_acc[NT][DVT];

    for (int j = 0; j < HPT; ++j) {
        __syncthreads();
        if (lane == 0) {
            sm_m[tile] = m[j];
        }
        __syncthreads();

        float m_b = -INFINITY;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            m_b = fmaxf(m_b, sm_m[t]);
        }

        const float e_t = m[j] == -INFINITY ? 0.0f : expf(m[j] - m_b);

        if (lane == 0) {
            sm_den[tile] = den[j]*e_t;
        }
#pragma unroll
        for (int e = 0; e < EPL2V; ++e) {
            const int i2 = lane + e*tile_sz;
            if (2*i2 < Dv) {
                sm_acc[tile][2*i2 + 0] = acc[j][2*e + 0]*e_t;
                sm_acc[tile][2*i2 + 1] = acc[j][2*e + 1]*e_t;
            }
        }
        __syncthreads();

        float * out = part + ((uint64_t)(qi*Hq + h0 + j)*nsplit + spl)*(2 + Dv);

        if (threadIdx.x == 0) {
            float den_b = 0.0f;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                den_b += sm_den[t];
            }
            out[0] = m_b;
            out[1] = den_b;
        }
        for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
            float a = 0.0f;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                a += sm_acc[t][d];
            }
            out[2 + d] = a;
        }
    }
}


// LDS-staged variant (ablation: env GGML_LOD_SPLIT=lds): one block per (query, KV head)
// covering the whole GQA group - each column's K/V is staged into shared memory ONCE
// and every wave (HPT query heads each) reads it from there, cutting global traffic by
// g/HPT vs the default kernel. Waves share the column stream, so there is no tile fold:
// every wave writes its heads' partials directly.
template <int tile_sz>
static __global__ void lod_attn_split_lds_f(
        const char * __restrict__ q,  const char * __restrict__ k,  const char * __restrict__ v,
        const char * __restrict__ ks, const char * __restrict__ vs,
        const int  * __restrict__ sel_in, const int * __restrict__ st,
        float      * __restrict__ part, // [n_rows, LOD_ATTN_SPLITS_LDS, 2 + Dv]
        const int Dk, const int Dv, const int Hq, const int Hkv,
        const int P, const int n_sel, const int ps,
        const uint64_t q_nb1,  const uint64_t q_nb2,
        const uint64_t k_nb1,  const uint64_t k_nb2,
        const uint64_t v_nb1,  const uint64_t v_nb2,
        const uint64_t ks_nb1, const uint64_t ks_nb2,
        const uint64_t vs_nb1, const uint64_t vs_nb2,
        const float scale, const float logn, const int k_f16, const int v_f16,
        const int sel_head) {
    const int qi  = blockIdx.x/Hkv;
    const int hk  = blockIdx.x%Hkv;
    const int spl = blockIdx.y;

    const int g    = Hq/Hkv;
    const int wave = threadIdx.x/tile_sz;
    const int lane = threadIdx.x%tile_sz;
    const int h0   = hk*g + wave*HPT; // this wave's first query head

    const int * sel = sel_in + (sel_head ? hk*n_sel : 0);

    const int prev_end = st[0];
    const int Pr       = st[1];
    const int limit    = prev_end + qi;
    const int tail0    = Pr*ps;
    const int n_tail   = limit - tail0 + 1;
    const int n_cols   = Pr + n_sel*ps + n_tail;

    const int epl2_k = (Dk/2 + tile_sz - 1)/tile_sz;
    const int epl2_v = (Dv/2 + tile_sz - 1)/tile_sz;

    extern __shared__ char smem_raw[];
    float    * sm_q   = (float *) smem_raw;          // [g][Dk]
    float    * kbuf   = sm_q + (size_t) g*Dk;        // [Dk]
    float    * vbuf   = kbuf + Dk;                   // [Dv]
    uint32_t * sm_sel = (uint32_t *)(vbuf + Dv);     // [(P+31)/32]

    for (int j = 0; j < g; ++j) {
        for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
            sm_q[j*Dk + d] = ((const float *)(q + qi*q_nb1 + (uint64_t)(hk*g + j)*q_nb2))[d];
        }
    }
    for (int i = threadIdx.x; i < (P + 31)/32; i += blockDim.x) {
        sm_sel[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        if (sel[i] >= 0) {
            atomicOr(&sm_sel[sel[i]/32], 1u << (sel[i]%32));
        }
    }
    __syncthreads();

    float m  [HPT];
    float den[HPT];
    float acc[HPT][LOD_ATTN_MAX_D/tile_sz] = {{ 0.0f }};
    for (int j = 0; j < HPT; ++j) {
        m[j]   = -INFINITY;
        den[j] = 0.0f;
    }

    for (int c = spl; c < n_cols; c += LOD_ATTN_SPLITS_LDS) {
        const char * kp;
        const char * vp;

        float bias    = 0.0f;
        float vscale  = 1.0f;
        float kscale  = scale;
        int   kf16    = k_f16;
        int   vf16    = v_f16;

        if (c < Pr) {
            if (sm_sel[c/32] & (1u << (c%32))) {
                continue; // uniform across the block
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
                continue;
            }
            const int col = pg*ps + (c - Pr)%ps;
            kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
            vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
        } else {
            const int col = tail0 + (c - Pr - n_sel*ps);
            kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
            vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
        }

        // stage this column's K and V once, converted to f32
        for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
            kbuf[d] = kf16 == 2 ? lod_load1_q8(kp, d) : kf16 ? __half2float(((const half *) kp)[d]) : ((const float *) kp)[d];
        }
        for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
            vbuf[d] = vf16 == 2 ? lod_load1_q8(vp, d) : vf16 ? __half2float(((const half *) vp)[d]) : ((const float *) vp)[d];
        }
        __syncthreads();

        float sj[HPT] = { 0.0f };
        for (int e = 0; e < epl2_k; ++e) {
            const int i2 = lane + e*tile_sz;
            if (2*i2 < Dk) {
                for (int j = 0; j < HPT; ++j) {
                    sj[j] += sm_q[(wave*HPT + j)*Dk + 2*i2]*kbuf[2*i2] + sm_q[(wave*HPT + j)*Dk + 2*i2 + 1]*kbuf[2*i2 + 1];
                }
            }
        }
#pragma unroll
        for (int off = tile_sz/2; off > 0; off >>= 1) {
            for (int j = 0; j < HPT; ++j) {
                sj[j] += __shfl_xor_sync(0xffffffff, sj[j], off, tile_sz);
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

        for (int e = 0; e < epl2_v; ++e) {
            const int i2 = lane + e*tile_sz;
            if (2*i2 < Dv) {
                for (int j = 0; j < HPT; ++j) {
                    acc[j][2*e + 0] += w[j]*vscale*vbuf[2*i2];
                    acc[j][2*e + 1] += w[j]*vscale*vbuf[2*i2 + 1];
                }
            }
        }
        __syncthreads(); // the stage buffers are rewritten next column
    }

    // no tile fold: every wave owns its heads' complete partials for this split
    for (int j = 0; j < HPT; ++j) {
        float * out = part + ((uint64_t)(qi*Hq + h0 + j)*LOD_ATTN_SPLITS_LDS + spl)*(2 + Dv);

        if (lane == 0) {
            out[0] = m[j];
            out[1] = m[j] == -INFINITY ? 0.0f : den[j];
        }
        for (int e = 0; e < epl2_v; ++e) {
            const int i2 = lane + e*tile_sz;
            if (2*i2 < Dv) {
                out[2 + 2*i2 + 0] = acc[j][2*e + 0];
                out[2 + 2*i2 + 1] = acc[j][2*e + 1];
            }
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
            ksd[d] += k_f16 == 2 ? lod_load1_q8(kc, d) : k_f16 ? __half2float(((const half *) kc)[d]) : ((const float *) kc)[d];
        }

        float       * vsd = (float *)((char *) vs + (uint64_t) page*vs_nb1 + hh*vs_nb2);
        const char  * vc  = v + (uint64_t) col*v_nb1 + hh*v_nb2;

        for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
            vsd[d] += v_f16 == 2 ? lod_load1_q8(vc, d) : v_f16 ? __half2float(((const half *) vc)[d]) : ((const float *) vc)[d];
        }
    }
}

// in-op selection, stage 1: pooled page scores, plus the sums fold in the same launch.
// blocks < P*Hkv score one (page, KV head) each (max over queries and group heads of
// dot(q, k_sums); the ranking is scale-invariant so the raw sums are used); the last
// Hkv blocks run the fold of lod_attn_fold_sums_f (fold writes pages >= Pr, scoring
// reads pages < Pr - disjoint)
template <int tile_sz>
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
                ksd[d] += k_f16 == 2 ? lod_load1_q8(kc, d) : k_f16 ? __half2float(((const half *) kc)[d]) : ((const float *) kc)[d];
            }

            float      * vsd = (float *)((char *) vsm + (uint64_t) page*vs_nb1 + hh*vs_nb2);
            const char * vc  = v + (uint64_t) col*v_nb1 + hh*v_nb2;

            for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
                vsd[d] += v_f16 == 2 ? lod_load1_q8(vc, d) : v_f16 ? __half2float(((const half *) vc)[d]) : ((const float *) vc)[d];
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

    float ksd[LOD_ATTN_MAX_D/tile_sz];
    for (int e = 0; e < Dk/tile_sz; ++e) {
        ksd[e] = ksp[lane + e*tile_sz];
    }

    float best = -INFINITY;
    for (int j = 0; j < g; ++j) {
        for (int qi = 0; qi < nq; ++qi) {
            const float * qp = (const float *)(q + (uint64_t) qi*q_nb1 + (uint64_t)(kv*g + j)*q_nb2);

            float s = 0.0f;
            for (int e = 0; e < Dk/tile_sz; ++e) {
                s += qp[lane + e*tile_sz]*ksd[e];
            }
#pragma unroll
            for (int off = tile_sz/2; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffff, s, off, tile_sz);
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
        const int * __restrict__ st, const int n_top, const int P, const int Hkv,
        const int sel_head) {
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
            if (sel_head) {
                v = scores[(uint64_t) blockIdx.x*P + i]; // this block's KV head row
            } else {
                for (int kv = 0; kv < Hkv; ++kv) {
                    v = fmaxf(v, scores[(uint64_t) kv*P + i]);
                }
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
        sel[blockIdx.x*n_top + i] = i < Pr ? si[i] : -1;
    }
}

// fallback above the sort capacity: iterative extraction
static __global__ void lod_attn_topk(
        float * __restrict__ scores, int * __restrict__ sel,
        const int * __restrict__ st, const int n_top, const int P, const int Hkv,
        const int sel_head) {
    const int Pr = st[1];

    __shared__ float sm_v[256];
    __shared__ int   sm_i[256];

    float * row = scores + (uint64_t) blockIdx.x*P; // this block's set (row 0 when shared)

    if (!sel_head) {
        for (int i = threadIdx.x; i < Pr; i += blockDim.x) {
            float v = row[i];
            for (int kv = 1; kv < Hkv; ++kv) {
                v = fmaxf(v, scores[(uint64_t) kv*P + i]);
            }
            row[i] = v;
        }
        __syncthreads();
    }

    for (int it = 0; it < n_top; ++it) {
        float bv = -INFINITY;
        int   bi = -1;
        for (int p = threadIdx.x; p < Pr; p += blockDim.x) {
            if (row[p] > bv) {
                bv = row[p];
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
            sel[blockIdx.x*n_top + it] = sm_i[0];
            if (sm_i[0] >= 0) {
                row[sm_i[0]] = -INFINITY;
            }
        }
        __syncthreads();
    }
}

// grid (n_rows, Dv/MERGE_SLICE): every block redundantly folds the tiny split
// headers, then combines its own slice of the value dimension - enough blocks
// to cover the whole GPU instead of one block per row
#define LOD_ATTN_MERGE_SLICE 64

template <int nsplit>
static __global__ void lod_attn_merge_f(
        const float * __restrict__ part, float * __restrict__ dst,
        const int Dv, const int Hq) {
    const int row = blockIdx.x; // qi*Hq + h
    const int d0  = blockIdx.y*LOD_ATTN_MERGE_SLICE;

    const float * p = part + (uint64_t) row*nsplit*(2 + Dv);

    __shared__ float sm_w[LOD_ATTN_SPLITS_LDS];

    float m = -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        m = fmaxf(m, p[s*(2 + Dv)]);
    }

    for (int s = threadIdx.x; s < nsplit; s += blockDim.x) {
        const float ms = p[s*(2 + Dv)];
        sm_w[s] = ms > -INFINITY/2 ? expf(ms - m) : 0.0f;
    }
    __syncthreads();

    // block-wide denominator: every thread re-sums the shared weights
    float den = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        den += p[s*(2 + Dv) + 1]*sm_w[s];
    }

    const float id = den > 0.0f ? 1.0f/den : 0.0f;

    // dst layout [Dv, Hq, nq]: row = qi*Hq + h maps directly
    float * out = dst + (uint64_t) row*Dv;

    for (int d = d0 + threadIdx.x; d < min(d0 + LOD_ATTN_MERGE_SLICE, Dv); d += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            const float w = sm_w[s];
            if (w > 0.0f) {
                a += p[s*(2 + Dv) + 2 + d]*w;
            }
        }
        out[d] = a*id;
    }
}


// mega-kernel variant (ablation: GGML_LOD_SPLIT=mega, wave64 only): the whole decode
// read - page scores + sums fold, top-k, split walk, merge - in ONE persistent launch.
// Phases are separated by a software grid barrier (monotonic arrive/release counters,
// no reset), so the launch is a plain <<<>>> and stays HIP-graph capturable; the host
// clamps the grid to the co-resident block capacity to make the barrier deadlock-free.
#define LOD_MEGA_SPLITS  32
#define LOD_MEGA_BLOCK  256

static __device__ void lod_mega_barrier(int * cnt, int * s_gen, const int nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int gen = *s_gen;
        if (atomicAdd(&cnt[0], 1) + 1 == nblocks*(gen + 1)) {
            atomicAdd(&cnt[1], 1); // last block releases the generation
        }
        while (atomicAdd(&cnt[1], 0) <= gen) {
        }
        *s_gen = gen + 1;
    }
    __syncthreads();
}

static __global__ void lod_attn_mega_f(
        const char * __restrict__ q,  const char * __restrict__ k,  const char * __restrict__ v,
        const char * __restrict__ ks, const char * __restrict__ vs,
        const int * __restrict__ st,
        float * __restrict__ scores, int * __restrict__ sel_ws, int * __restrict__ bar,
        float * __restrict__ part, float * __restrict__ dstf,
        const int Dk, const int Dv, const int nq, const int Hq, const int Hkv,
        const int P, const int n_top, const int ps,
        const uint64_t q_nb1,  const uint64_t q_nb2,
        const uint64_t k_nb1,  const uint64_t k_nb2,
        const uint64_t v_nb1,  const uint64_t v_nb2,
        const uint64_t ks_nb1, const uint64_t ks_nb2,
        const uint64_t vs_nb1, const uint64_t vs_nb2,
        const float scale, const float logn, const int k_f16, const int v_f16,
        const int sel_head) {
    constexpr int TS = 64;

    const int g      = Hq/Hkv;
    const int wave   = threadIdx.x/TS;
    const int lane   = threadIdx.x%TS;
    const int gwaves = gridDim.x*(LOD_MEGA_BLOCK/TS);
    const int gwave  = blockIdx.x*(LOD_MEGA_BLOCK/TS) + wave;

    const int prev_end = st[0];
    const int Pr       = st[1];

    __shared__ int s_gen;
    if (threadIdx.x == 0) {
        s_gen = 0;
    }

    // ---- phase A: page scores (one (page, kv) item per wave) + sums fold ----
    for (int it = gwave; it < P*Hkv + Hkv; it += gwaves) {
        if (it >= P*Hkv) { // fold: one KV head per wave
            const int hh = it - P*Hkv;
            for (int t = 0; t < nq; ++t) {
                const int col  = prev_end + t;
                const int page = col/ps;
                float       * ksd = (float *)((char *)(void *) ks + (uint64_t) page*ks_nb1 + hh*ks_nb2);
                const char  * kc  = k + (uint64_t) col*k_nb1 + hh*k_nb2;
                for (int d = lane; d < Dk; d += TS) {
                    ksd[d] += k_f16 == 2 ? lod_load1_q8(kc, d) : k_f16 ? __half2float(((const half *) kc)[d]) : ((const float *) kc)[d];
                }
                float       * vsd = (float *)((char *)(void *) vs + (uint64_t) page*vs_nb1 + hh*vs_nb2);
                const char  * vc  = v + (uint64_t) col*v_nb1 + hh*v_nb2;
                for (int d = lane; d < Dv; d += TS) {
                    vsd[d] += v_f16 == 2 ? lod_load1_q8(vc, d) : v_f16 ? __half2float(((const half *) vc)[d]) : ((const float *) vc)[d];
                }
            }
            continue;
        }
        const int p  = it/Hkv;
        const int kv = it%Hkv;
        if (p >= Pr) {
            continue;
        }
        const float * ksp = (const float *)(ks + (uint64_t) p*ks_nb1 + (uint64_t) kv*ks_nb2);
        float ksd[LOD_ATTN_MAX_D/TS];
        for (int e = 0; e < Dk/TS; ++e) {
            ksd[e] = ksp[lane + e*TS];
        }
        float best = -INFINITY;
        for (int j = 0; j < g; ++j) {
            for (int qi = 0; qi < nq; ++qi) {
                const float * qp = (const float *)(q + (uint64_t) qi*q_nb1 + (uint64_t)(kv*g + j)*q_nb2);
                float sc = 0.0f;
                for (int e = 0; e < Dk/TS; ++e) {
                    sc += qp[lane + e*TS]*ksd[e];
                }
#pragma unroll
                for (int off = TS/2; off > 0; off >>= 1) {
                    sc += __shfl_xor_sync(0xffffffff, sc, off, TS);
                }
                best = fmaxf(best, sc);
            }
        }
        if (lane == 0) {
            scores[(uint64_t) kv*P + p] = best;
        }
    }

    lod_mega_barrier(bar, &s_gen, gridDim.x);

    // ---- phase B: top-k (block 0, or blocks 0..Hkv-1 with per-head sets) ----
    const int n_sets = sel_head ? Hkv : 1;
    if ((int) blockIdx.x < n_sets && P <= LOD_TOPK_SORT_MAX) {
        __shared__ float sv[LOD_TOPK_SORT_MAX];
        __shared__ int   si[LOD_TOPK_SORT_MAX];
        int n = 1;
        while (n < Pr) {
            n <<= 1;
        }
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float vv = -INFINITY;
            if (i < Pr) {
                if (sel_head) {
                    vv = scores[(uint64_t) blockIdx.x*P + i];
                } else {
                    for (int kv = 0; kv < Hkv; ++kv) {
                        vv = fmaxf(vv, scores[(uint64_t) kv*P + i]);
                    }
                }
            }
            sv[i] = vv;
            si[i] = i;
        }
        __syncthreads();
        for (int ksz = 2; ksz <= n; ksz <<= 1) {
            for (int j = ksz >> 1; j > 0; j >>= 1) {
                for (int i = threadIdx.x; i < n; i += blockDim.x) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
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
            sel_ws[blockIdx.x*n_top + i] = i < Pr ? si[i] : -1;
        }
    }

    lod_mega_barrier(bar, &s_gen, gridDim.x);

    // ---- phase C: split walk (one (head-pair-row, split) item per block) ----
    {
        const int nhp   = Hq/HPT;
        const int nitem = nq*nhp*LOD_MEGA_SPLITS;
        const int tile  = wave; // 4 tiles of one wave each

        for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
            const int row = it%(nq*nhp);
            const int spl = it/(nq*nhp);
            const int qi  = row/nhp;
            const int h0  = (row%nhp)*HPT;
            const int hk  = h0/g;

            const int * sel = sel_ws + (sel_head ? hk*n_top : 0);

            const int limit  = prev_end + qi;
            const int tail0  = Pr*ps;
            const int n_cols = Pr + n_top*ps + (limit - tail0 + 1);

            const int epl2_k = (Dk/2 + TS - 1)/TS;
            const int epl2_v = (Dv/2 + TS - 1)/TS;

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
            for (int i = threadIdx.x; i < n_top; i += blockDim.x) {
                if (sel[i] >= 0) {
                    atomicOr(&sm_sel[sel[i]/32], 1u << (sel[i]%32));
                }
            }
            __syncthreads();

            float m  [HPT];
            float den[HPT];
            float acc[HPT][LOD_ATTN_MAX_D/TS] = {{ 0.0f }};
            for (int j = 0; j < HPT; ++j) {
                m[j]   = -INFINITY;
                den[j] = 0.0f;
            }

            for (int c = spl*LOD_ATTN_NTILES + tile; c < n_cols; c += LOD_MEGA_SPLITS*LOD_ATTN_NTILES) {
                const char * kp;
                const char * vp;
                float bias = 0.0f, vscale = 1.0f, kscale = scale;
                int kf16 = k_f16, vf16 = v_f16;
                if (c < Pr) {
                    if (sm_sel[c/32] & (1u << (c%32))) {
                        continue;
                    }
                    kp = ks + (uint64_t) c*ks_nb1 + hk*ks_nb2;
                    vp = vs + (uint64_t) c*vs_nb1 + hk*vs_nb2;
                    bias = logn; vscale = 1.0f/ps; kscale = scale/ps; kf16 = 0; vf16 = 0;
                } else if (c < Pr + n_top*ps) {
                    const int pg = sel[(c - Pr)/ps];
                    if (pg < 0) {
                        continue;
                    }
                    const int col = pg*ps + (c - Pr)%ps;
                    kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
                    vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
                } else {
                    const int col = tail0 + (c - Pr - n_top*ps);
                    kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
                    vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
                }
                float sj[HPT] = { 0.0f };
                if (kf16) {
                    const half2 * kh = (const half2 *) kp;
                    for (int e = 0; e < epl2_k; ++e) {
                        const int i2 = lane + e*TS;
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
                        const int i2 = lane + e*TS;
                        if (2*i2 < Dk) {
                            const float2 kv2 = kf[i2];
                            for (int j = 0; j < HPT; ++j) {
                                sj[j] += sm_q[j][2*i2]*kv2.x + sm_q[j][2*i2 + 1]*kv2.y;
                            }
                        }
                    }
                }
#pragma unroll
                for (int off = TS/2; off > 0; off >>= 1) {
                    for (int j = 0; j < HPT; ++j) {
                        sj[j] += __shfl_xor_sync(0xffffffff, sj[j], off, TS);
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
                        const int i2 = lane + e*TS;
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
                        const int i2 = lane + e*TS;
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

            // fold the 4 tiles of this block, one head at a time (same as the default kernel)
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
                    const int i2 = lane + e*TS;
                    if (2*i2 < Dv) {
                        sm_acc[tile][2*i2 + 0] = acc[j][2*e + 0]*e_t;
                        sm_acc[tile][2*i2 + 1] = acc[j][2*e + 1]*e_t;
                    }
                }
                __syncthreads();
                float * out = part + ((uint64_t)(qi*Hq + h0 + j)*LOD_MEGA_SPLITS + spl)*(2 + Dv);
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
            __syncthreads();
        }
    }

    lod_mega_barrier(bar, &s_gen, gridDim.x);

    // ---- phase D: merge (one (row, d-slice) item per block, first wave active) ----
    {
        const int nslice = (Dv + TS - 1)/TS;
        for (int it = blockIdx.x; it < nq*Hq*nslice; it += gridDim.x) {
            if (wave != 0) {
                continue;
            }
            const int row = it/nslice;
            const int d0  = (it%nslice)*TS;
            const float * pp = part + (uint64_t) row*LOD_MEGA_SPLITS*(2 + Dv);
            float mx = -INFINITY;
            for (int sp = 0; sp < LOD_MEGA_SPLITS; ++sp) {
                mx = fmaxf(mx, pp[sp*(2 + Dv)]);
            }
            float den = 0.0f;
            float a   = 0.0f;
            const int d = d0 + lane;
            for (int sp = 0; sp < LOD_MEGA_SPLITS; ++sp) {
                const float ms = pp[sp*(2 + Dv)];
                if (ms > -INFINITY/2) {
                    const float w = expf(ms - mx);
                    den += pp[sp*(2 + Dv) + 1]*w;
                    if (d < Dv) {
                        a += pp[sp*(2 + Dv) + 2 + d]*w;
                    }
                }
            }
            const float id = den > 0.0f ? 1.0f/den : 0.0f; // reciprocal-multiply, bitwise-matching the merge kernel
            if (d < Dv) {
                dstf[(uint64_t) row*Dv + d] = a*id;
            }
        }
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
           q->ne[0]%64 == 0 && // in-op selection: lanes split the head dim evenly (either tile size)

           ks->ne[1] <= LOD_ATTN_MAX_P &&
           (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32 ||
                (k->type == GGML_TYPE_Q8_0 && q->ne[0]%QK8_0 == 0)) &&
           (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32 ||
                (v->type == GGML_TYPE_Q8_0 && v->ne[0]%QK8_0 == 0)) &&
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

    // one tile = one hardware wave (64 on CDNA, 32 on RDNA/NVIDIA)
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // ablation switch: GGML_LOD_SPLIT=lds stages K/V through shared memory with one
    // block per (query, KV head); default is the per-head-pair kernel
    static const char * split_env = getenv("GGML_LOD_SPLIT");
    const int g = Hq/Hkv;
    const size_t lds_smem = ((size_t) g*Dk + Dk + Dv)*sizeof(float) + (size_t)((P + 31)/32)*sizeof(uint32_t);
    const bool use_lds = split_env != nullptr && strcmp(split_env, "lds") == 0 &&
            g%HPT == 0 && (g/HPT)*256 <= 1024*4 && lds_smem <= 48*1024;

    // split-K width: the read is latency bound, so the split count is picked to keep
    // roughly LOD_ATTN_BPC blocks per compute unit resident. Splitting further only
    // widens the merge (measured: 128 splits lose on every geometry).
    // GGML_LOD_SPLITS=N overrides it (ablation).
    static const char * splits_env = getenv("GGML_LOD_SPLITS");

    int nsplit = LOD_ATTN_SPLITS;
    if (!use_lds) {
        const int nrb = std::max(1, (nq*Hq)/HPT); // row blocks already in the grid
        const int want = LOD_ATTN_BPC*ggml_cuda_info().devices[ctx.device].nsm / nrb;

        nsplit = want >= 64 ? 64 : want >= 32 ? 32 : 16;
    } else {
        nsplit = LOD_ATTN_SPLITS_LDS;
    }
    if (splits_env != nullptr) {
        const int e = atoi(splits_env);
        nsplit = (e == 16 || e == 32 || e == 64 || e == 128) ? e : nsplit;
    }

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) n_rows*nsplit*(2 + Dv));

    const int k_f16 = k->type == GGML_TYPE_Q8_0 ? 2 : k->type == GGML_TYPE_F16 ? 1 : 0;
    const int v_f16 = v->type == GGML_TYPE_Q8_0 ? 2 : v->type == GGML_TYPE_F16 ? 1 : 0;

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

        const int sel_head = ggml_get_op_params_i32(dst, 3);
        const int n_sets   = sel_head ? Hkv : 1;

        ws_scores.alloc((size_t) P*Hkv);
        ws_sel.alloc((size_t) n_sel*n_sets);

        // ablation: GGML_LOD_SPLIT=mega runs the whole read in one persistent launch
        // (in-op selection, wave64, bitonic-capable page count only)
            if (split_env != nullptr && strcmp(split_env, "mega") == 0 &&
                    warp_size == 64 && Dk%64 == 0 && P <= LOD_TOPK_SORT_MAX && g%HPT == 0 &&
                    k_f16 < 2 && v_f16 < 2) {
            ggml_cuda_pool_alloc<int32_t> bar(ctx.pool(), 2);
            CUDA_CHECK(cudaMemsetAsync(bar.get(), 0, 2*sizeof(int32_t), stream));

            int maxb = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, lod_attn_mega_f, LOD_MEGA_BLOCK, 0));
            const int G = std::min(std::max(maxb, 1)*ggml_cuda_info().devices[ctx.device].nsm, 1024);

            lod_attn_mega_f<<<G, LOD_MEGA_BLOCK, 0, stream>>>(
                    (const char *) q->data, (const char *) k->data, (const char *) v->data,
                    (const char *) ks->data, (const char *) vs->data,
                    (const int *) st->data,
                    ws_scores.get(), ws_sel.get(), bar.get(),
                    part.get(), (float *) dst->data,
                    Dk, Dv, nq, Hq, Hkv, P, n_sel, ps,
                    q->nb[1],  q->nb[2],  k->nb[1],  k->nb[2],  v->nb[1], v->nb[2],
                    ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
                    scale, logf((float) ps), k_f16, v_f16, sel_head ? 1 : 0);
            return;
        }

        const auto launch_score = [&](auto tile_tag) {
            constexpr int TS = decltype(tile_tag)::value;
            lod_attn_score_fold_f<TS><<<P*Hkv + Hkv, TS, 0, stream>>>(
                    (const char *) q->data, (const char *) ks->data,
                    (const char *) k->data, (const char *) v->data,
                    (float *) ks->data, (float *) vs->data, (const int *) st->data,
                    ws_scores.get(),
                    Dk, Dv, nq, Hq, Hkv, P, ps,
                    q->nb[1], q->nb[2],
                    ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
                    k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                    k_f16, v_f16);
        };
        if (warp_size == 64) {
            launch_score(std::integral_constant<int, 64>{});
        } else {
            launch_score(std::integral_constant<int, 32>{});
        }

        if (P <= LOD_TOPK_SORT_MAX) {
            lod_attn_topk_sort<<<n_sets, 256, 0, stream>>>(ws_scores.get(), ws_sel.get(), (const int *) st->data, n_sel, P, Hkv, sel_head);
        } else {
            lod_attn_topk<<<n_sets, 256, 0, stream>>>(ws_scores.get(), ws_sel.get(), (const int *) st->data, n_sel, P, Hkv, sel_head);
        }

        sel_ptr = ws_sel.get();

        // temp diagnostics: env LLAMA_LOD_DBG_SCORES="p1,p2,..." prints the selection
        // rank of the listed pages (max-pooled over KV heads) per call, decode only
        static const char * dbg_env = getenv("LLAMA_LOD_DBG_SCORES");
        if (dbg_env != nullptr && nq == 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float>   hs((size_t) P*Hkv);
            std::vector<int32_t> hst(2);
            CUDA_CHECK(cudaMemcpy(hs.data(), ws_scores.get(), hs.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hst.data(), st->data, 2*sizeof(int32_t), cudaMemcpyDeviceToHost));
            const int Pr = hst[1];
            std::vector<float> pooled(Pr, -INFINITY);
            for (int kv = 0; kv < Hkv; ++kv) {
                for (int p2 = 0; p2 < Pr; ++p2) {
                    pooled[p2] = std::max(pooled[p2], hs[(size_t) kv*P + p2]);
                }
            }
            fprintf(stderr, "LODRANK %s n_past=%d Pr=%d:", dst->name, hst[0], Pr);
            const char * c = dbg_env;
            while (*c) {
                const int pg = atoi(c);
                if (pg >= 0 && pg < Pr) {
                    int rank = 0;
                    for (int p2 = 0; p2 < Pr; ++p2) {
                        if (pooled[p2] > pooled[pg]) {
                            rank++;
                        }
                    }
                    fprintf(stderr, " p%d=r%d", pg, rank);
                }
                while (*c && *c != ',') c++;
                if (*c == ',') c++;
            }
            fprintf(stderr, "\n");
        }
    }


    const int op_sel_head = sel == nullptr ? ggml_get_op_params_i32(dst, 3) : 0;

    const auto launch_split = [&](auto tile_tag) {
        constexpr int TS = decltype(tile_tag)::value;

        if (use_lds) {
            const dim3 grid(nq*Hkv, LOD_ATTN_SPLITS_LDS, 1);
            const dim3 block((g/HPT)*TS, 1, 1);

            lod_attn_split_lds_f<TS><<<grid, block, lds_smem, stream>>>(
                    (const char *) q->data, (const char *) k->data, (const char *) v->data,
                    (const char *) ks->data, (const char *) vs->data,
                    sel_ptr, (const int *) st->data,
                    part.get(),
                    Dk, Dv, Hq, Hkv, P, n_sel, ps,
                    q->nb[1],  q->nb[2],  k->nb[1],  k->nb[2],  v->nb[1], v->nb[2],
                    ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
                    scale, logf((float) ps), k_f16, v_f16,
                    op_sel_head);
        } else {
            const dim3 grid(n_rows/HPT, nsplit, 1);
            const dim3 block(LOD_ATTN_BLOCK, 1, 1);

            // instantiate the accumulator width the head dim actually needs
            const auto launch_ev = [&](auto ev_tag) {
                constexpr int EV = decltype(ev_tag)::value;

                lod_attn_split_f<TS, EV><<<grid, block, 0, stream>>>(
                        (const char *) q->data, (const char *) k->data, (const char *) v->data,
                        (const char *) ks->data, (const char *) vs->data,
                        sel_ptr, (const int *) st->data,
                        part.get(),
                        Dk, Dv, Hq, Hkv, P, n_sel, ps,
                        q->nb[1],  q->nb[2],  k->nb[1],  k->nb[2],  v->nb[1], v->nb[2],
                        ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
                        scale, logf((float) ps), k_f16, v_f16,
                        op_sel_head, nsplit);
            };

            const int need = (Dv/2 + TS - 1)/TS;
            if (need <= 1) {
                launch_ev(std::integral_constant<int, 1>{});
            } else if (need <= 2) {
                launch_ev(std::integral_constant<int, 2>{});
            } else if (need <= 4) {
                launch_ev(std::integral_constant<int, 4>{});
            } else if constexpr (TS < 64) { // wave64 covers LOD_ATTN_MAX_D with 4 pairs
                launch_ev(std::integral_constant<int, 8>{});
            } else {
                GGML_ABORT("lod_attn: head dim too large for the split kernel");
            }
        }
    };
    if (warp_size == 64) {
        launch_split(std::integral_constant<int, 64>{});
    } else {
        launch_split(std::integral_constant<int, 32>{});
    }

    const dim3 grid_m(n_rows, (Dv + LOD_ATTN_MERGE_SLICE - 1)/LOD_ATTN_MERGE_SLICE, 1);

    switch (nsplit) {
        case LOD_ATTN_SPLITS_LDS:
            lod_attn_merge_f<LOD_ATTN_SPLITS_LDS><<<grid_m, LOD_ATTN_MERGE_SLICE, 0, stream>>>(part.get(), (float *) dst->data, Dv, Hq);
            break;
        case 64:
            lod_attn_merge_f<64><<<grid_m, LOD_ATTN_MERGE_SLICE, 0, stream>>>(part.get(), (float *) dst->data, Dv, Hq);
            break;
        case 16:
            lod_attn_merge_f<16><<<grid_m, LOD_ATTN_MERGE_SLICE, 0, stream>>>(part.get(), (float *) dst->data, Dv, Hq);
            break;
        default:
            lod_attn_merge_f<LOD_ATTN_SPLITS><<<grid_m, LOD_ATTN_MERGE_SLICE, 0, stream>>>(part.get(), (float *) dst->data, Dv, Hq);
            break;
    }
}
