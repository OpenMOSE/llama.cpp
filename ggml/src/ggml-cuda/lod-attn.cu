// LoD sparse attention read: one online softmax over three tiers
//   [unselected page summaries | leaves of the selected pages | raw tail + current tokens]
// split-K over a virtual column space, then a small merge kernel folds the partials.

#include "lod-attn.cuh"

#define LOD_ATTN_SPLITS   64
#define LOD_ATTN_TILE     32
#define LOD_ATTN_NTILES    4
#define LOD_ATTN_MAX_D   512
#define LOD_ATTN_MAX_P  8192

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
    const int row = blockIdx.x;
    const int spl = blockIdx.y;

    const int qi = row/Hq;
    const int h  = row%Hq;
    const int hk = h/(Hq/Hkv);

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

    __shared__ float    sm_q[LOD_ATTN_MAX_D];
    __shared__ uint32_t sm_sel[LOD_ATTN_MAX_P/32];

    for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
        sm_q[d] = ((const float *)(q + qi*q_nb1 + h*q_nb2))[d];
    }
    for (int i = threadIdx.x; i < (P + 31)/32; i += blockDim.x) {
        sm_sel[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        atomicOr(&sm_sel[sel[i]/32], 1u << (sel[i]%32));
    }
    __syncthreads();

    float m   = -INFINITY;
    float den = 0.0f;
    float acc[LOD_ATTN_MAX_D/LOD_ATTN_TILE] = { 0.0f }; // 2 consecutive elements per slot pair

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
            const int col = sel[(c - Pr)/ps]*ps + (c - Pr)%ps;
            kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
            vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
        } else {
            const int col = tail0 + (c - Pr - n_sel*ps);
            kp = k + (uint64_t) col*k_nb1 + hk*k_nb2;
            vp = v + (uint64_t) col*v_nb1 + hk*v_nb2;
        }

        // dot(q, k): two consecutive elements per lane per step
        float s = 0.0f;
        if (kf16) {
            const half2 * kh = (const half2 *) kp;
            for (int e = 0; e < epl2_k; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dk) {
                    const float2 kv2 = __half22float2(kh[i2]);
                    s += sm_q[2*i2]*kv2.x + sm_q[2*i2 + 1]*kv2.y;
                }
            }
        } else {
            const float2 * kf = (const float2 *) kp;
            for (int e = 0; e < epl2_k; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dk) {
                    const float2 kv2 = kf[i2];
                    s += sm_q[2*i2]*kv2.x + sm_q[2*i2 + 1]*kv2.y;
                }
            }
        }
#pragma unroll
        for (int off = LOD_ATTN_TILE/2; off > 0; off >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, off, LOD_ATTN_TILE);
        }

        const float logit = s*kscale + bias;

        float w;
        if (logit > m) {
            const float e = expf(m - logit);
            den *= e;
            for (int i = 0; i < 2*epl2_v; ++i) {
                acc[i] *= e;
            }
            m = logit;
            w = 1.0f;
        } else {
            w = expf(logit - m);
        }
        den += w;

        if (vf16) {
            const half2 * vh = (const half2 *) vp;
            for (int e = 0; e < epl2_v; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dv) {
                    const float2 vv2 = __half22float2(vh[i2]);
                    acc[2*e + 0] += w*vscale*vv2.x;
                    acc[2*e + 1] += w*vscale*vv2.y;
                }
            }
        } else {
            const float2 * vf = (const float2 *) vp;
            for (int e = 0; e < epl2_v; ++e) {
                const int i2 = lane + e*LOD_ATTN_TILE;
                if (2*i2 < Dv) {
                    const float2 vv2 = vf[i2];
                    acc[2*e + 0] += w*vscale*vv2.x;
                    acc[2*e + 1] += w*vscale*vv2.y;
                }
            }
        }
    }

    // fold the tiles of this block through shared memory
    __shared__ float sm_m  [LOD_ATTN_NTILES];
    __shared__ float sm_den[LOD_ATTN_NTILES];
    __shared__ float sm_acc[LOD_ATTN_NTILES][LOD_ATTN_MAX_D];

    if (lane == 0) {
        sm_m[tile] = m;
    }
    __syncthreads();

    float m_b = -INFINITY;
    for (int t = 0; t < LOD_ATTN_NTILES; ++t) {
        m_b = fmaxf(m_b, sm_m[t]);
    }

    const float e_t = m == -INFINITY ? 0.0f : expf(m - m_b);

    if (lane == 0) {
        sm_den[tile] = den*e_t;
    }
    for (int e = 0; e < epl2_v; ++e) {
        const int i2 = lane + e*LOD_ATTN_TILE;
        if (2*i2 < Dv) {
            sm_acc[tile][2*i2 + 0] = acc[2*e + 0]*e_t;
            sm_acc[tile][2*i2 + 1] = acc[2*e + 1]*e_t;
        }
    }
    __syncthreads();

    float * out = part + ((uint64_t) row*LOD_ATTN_SPLITS + spl)*(2 + Dv);

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

static __global__ void lod_attn_merge_f(
        const float * __restrict__ part, float * __restrict__ dst,
        const int Dv, const int Hq) {
    const int row = blockIdx.x; // qi*Hq + h

    const float * p = part + (uint64_t) row*LOD_ATTN_SPLITS*(2 + Dv);

    float m = -INFINITY;
    for (int s = 0; s < LOD_ATTN_SPLITS; ++s) {
        m = fmaxf(m, p[s*(2 + Dv)]);
    }

    float den = 0.0f;
    for (int s = 0; s < LOD_ATTN_SPLITS; ++s) {
        const float ms = p[s*(2 + Dv)];
        if (ms > -INFINITY/2) {
            den += p[s*(2 + Dv) + 1]*expf(ms - m);
        }
    }

    const float id = den > 0.0f ? 1.0f/den : 0.0f;

    // dst layout [Dv, Hq, nq]: row = qi*Hq + h maps directly
    float * out = dst + (uint64_t) row*Dv;

    for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < LOD_ATTN_SPLITS; ++s) {
            const float ms = p[s*(2 + Dv)];
            if (ms > -INFINITY/2) {
                a += p[s*(2 + Dv) + 2 + d]*expf(ms - m);
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

    lod_attn_fold_sums_f<<<Hkv, 256, 0, stream>>>(
            (const char *) k->data, (const char *) v->data,
            (float *) ks->data, (float *) vs->data, (const int *) st->data,
            Dk, Dv, nq, ps,
            k->nb[1], k->nb[2], v->nb[1], v->nb[2],
            ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
            k->type == GGML_TYPE_F16, v->type == GGML_TYPE_F16);

    const dim3 grid(n_rows, LOD_ATTN_SPLITS, 1);
    const dim3 block(LOD_ATTN_TILE*LOD_ATTN_NTILES, 1, 1);

    lod_attn_split_f<<<grid, block, 0, stream>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const char *) ks->data, (const char *) vs->data,
            (const int *) sel->data, (const int *) st->data,
            part.get(),
            Dk, Dv, Hq, Hkv, P, (int) sel->ne[0], ps,
            q->nb[1],  q->nb[2],  k->nb[1],  k->nb[2],  v->nb[1], v->nb[2],
            ks->nb[1], ks->nb[2], vs->nb[1], vs->nb[2],
            scale, logf((float) ps), k->type == GGML_TYPE_F16, v->type == GGML_TYPE_F16);

    lod_attn_merge_f<<<n_rows, 256, 0, stream>>>(part.get(), (float *) dst->data, Dv, Hq);
}
