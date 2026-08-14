// LoD2 attention (ref. "Key-Value Means", arXiv:2605.09877) on CUDA/HIP.
//
// The CPU implementation in ggml/src/ggml-cpu/lod2.cpp is the contract; this
// file must agree with it (tests/test-lod2.cpp checks both against the same
// oracle).  Two things are borrowed from the author's Triton kernels:
//
//   * softmax in base two - every logit carries a factor log2(e) so the inner
//     loop uses exp2f/log2f instead of expf/logf.  This is an exact change of
//     base: softmax(log2(e)*x) with exp2 equals softmax(x) with exp, and the
//     branch logsumexps scale by the same factor, so the LSE merge is unchanged.
//   * the running (max, denominator, accumulator) online softmax, so no branch
//     ever materializes its score vector.
//
// Unlike the Triton kernels the state update here is deterministic: merges are
// accumulated by one block per slot in ascending token order instead of with
// atomics, so repeated runs give bit-identical state.  Page ids still come from
// an atomic bump allocator, but a page id is a label that nothing compares.

#include "lod2.cuh"

#include <cfloat>

// ggml hardcodes WARP_SIZE to 32; on CDNA the wave is 64, so a "warp" there is
// a half wave and every lane carries twice the value elements it needs to.
// These kernels keep their own lane count and shuffle at that width.
#ifdef GGML_USE_HIP
#define LOD2_LANES 64
#else
#define LOD2_LANES WARP_SIZE
#endif

#define LOD2_BLOCK   256
#define LOD2_WPB     (LOD2_BLOCK/LOD2_LANES)   // warps per block
#define LOD2_RED_BLOCK 1024                    // lod2_k_reduce runs 3*Dv threads
#define LOD2_SB        8   // slots staged in shared memory per round
#define LOD2_MAX_ROW 512   // max D + Dv the staging buffer holds
#define LOD2_QT        8   // local columns scored per tile in lod2_k_attn
#define LOD2_MAX_R     8   // routes, capped by the algorithm
#define LOD2_TOPK_N  (LOD2_MAX_R + 1)          // top-R plus the coarse branch's own max
#define LOD2_LOG2E  1.4426950408889634f

// K/V element types the kernels read directly
#define LOD2_T_F32  0
#define LOD2_T_F16  1
#define LOD2_T_Q8_0 2

// The KV archive is read one element at a time, so a quantized cache only needs
// its dequantisation here - the shared-memory tiles downstream are f32 and do
// not care where the value came from.  Without this the scheduler tries to move
// LOD2_UPDATE to the CPU, and the state tensors are pre-allocated in the device
// buffer, so ggml_backend_sched aborts instead of falling back.
static __device__ __forceinline__ float lod2_get(const char * p, const int type, const int i) {
    switch (type) {
        case LOD2_T_F32:  return ((const float *) p)[i];
        case LOD2_T_Q8_0: {
            const block_q8_0 * b = (const block_q8_0 *) p + i/QK8_0;
            return __half2float(b->d)*(float) b->qs[i % QK8_0];
        }
        default:          return __half2float(((const half *) p)[i]);
    }
}

template <int L>
static __device__ __forceinline__ float lod2_wsum(float x) {
#pragma unroll
    for (int m = L/2; m > 0; m >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, m, L);
    }
    return x;
}

static __device__ __forceinline__ float lod2_warp_sum(float x) {
    return lod2_wsum<LOD2_LANES>(x);
}

// Load W consecutive floats per lane.  The coarse pass streams the value half
// of the mean table and nothing else, so it wants the widest load the head size
// allows: one dwordx4 per lane covers a 256-wide head in a single instruction
// where the lane-strided form needed four.
template <int W>
static __device__ __forceinline__ void lod2_ldw(float * dst, const float * src, const bool ok) {
    if (W == 4) {
        const float4 t = ok ? *(const float4 *) src : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        dst[0] = t.x; dst[1] = t.y; dst[2] = t.z; dst[3] = t.w;
    } else if (W == 2) {
        const float2 t = ok ? *(const float2 *) src : make_float2(0.0f, 0.0f);
        dst[0] = t.x; dst[1] = t.y;
    } else {
        dst[0] = ok ? *src : 0.0f;
    }
}

// max with the lowest index winning ties, so CPU and GPU pick the same slot
static __device__ __forceinline__ void lod2_warp_argmax(float & val, int & idx) {
#pragma unroll
    for (int m = LOD2_LANES/2; m > 0; m >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, val, m, LOD2_LANES);
        const int   i = __shfl_xor_sync(0xffffffff, idx, m, LOD2_LANES);
        if (v > val || (v == val && i < idx)) {
            val = v;
            idx = i;
        }
    }
}

// ---------------------------------------------------------------------------
// state update

// 1. how novel is each new column, and where would it merge
//
// MT columns share one pass over the slot table.  That table is the whole cost
// of this kernel: with one block per column, the same 3 MB per head at 32k came
// back once per absorbed column, and a state update absorbs 256 of them at
// generation time and 1280 at prefill.  A column's score depends on nothing but
// itself and the slot, so the read is shared simply by staging MT columns.
// Each (column, slot) dot keeps its own lane-strided accumulation and the same
// warp reduction, and the block merge is a max by (score, lowest slot) either
// way, so the scores and the winners are unchanged.
template <int MT>
static __global__ void lod2_k_sim(
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const float * __restrict__ s_kv,
        const int S_cap, const int Ds, const int D, const int Dv,
        const int p0, const int M, const int s_len, const int sink,
        float * __restrict__ best_score, int * __restrict__ best_slot, float * __restrict__ sel) {
    const int m0 = blockIdx.x*MT;
    const int h  = blockIdx.y;

    const int lane  = threadIdx.x % LOD2_LANES;
    const int warp  = threadIdx.x / LOD2_LANES;
    const int nwarp = blockDim.x / LOD2_LANES;

    extern __shared__ float smem[];
    float * kq  = smem;                          // [MT][D]
    float * shb = kq  + (size_t) MT*D;           // [MT][nwarp]
    float * shp = shb + (size_t) MT*nwarp;
    int   * sha = (int *)(shp + (size_t) MT*nwarp);

#pragma unroll
    for (int u = 0; u < MT; ++u) {
        // a tail block stages a column it will not report, so the dot never
        // sees uninitialised shared memory
        const char * krow = k + (size_t)(p0 + min(m0 + u, M - 1))*knb1 + (size_t) h*knb2;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            kq[u*D + i] = lod2_get(krow, ktype, i);
        }
    }
    __syncthreads();

    const float * slot = s_kv + (size_t) h*S_cap*Ds;

    float wbest[MT], wprot[MT];
    int   warg [MT];
#pragma unroll
    for (int u = 0; u < MT; ++u) {
        wbest[u] = -INFINITY;
        wprot[u] = -INFINITY;
        warg [u] = INT_MAX;
    }

    for (int s = warp; s < s_len; s += nwarp) {
        const float * row = slot + (size_t) s*Ds;
        const float cnt = row[D + Dv];
        if (!(cnt > 0.5f)) {
            continue;
        }
        float acc[MT];
#pragma unroll
        for (int u = 0; u < MT; ++u) {
            acc[u] = 0.0f;
        }
        for (int i = lane; i < D; i += LOD2_LANES) {
            const float x = row[i];
#pragma unroll
            for (int u = 0; u < MT; ++u) {
                acc[u] += kq[u*D + i]*x;
            }
        }
#pragma unroll
        for (int u = 0; u < MT; ++u) {
            const float a = lod2_warp_sum(acc[u])/cnt;
            if (s < sink) {
                wprot[u] = fmaxf(wprot[u], a);
            } else if (a > wbest[u] || (a == wbest[u] && s < warg[u])) {
                wbest[u] = a;
                warg [u] = s;
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < MT; ++u) {
            shb[u*nwarp + warp] = wbest[u];
            shp[u*nwarp + warp] = wprot[u];
            sha[u*nwarp + warp] = warg [u];
        }
    }
    __syncthreads();

    for (int u = threadIdx.x; u < MT; u += blockDim.x) {
        if (m0 + u >= M) {
            continue;
        }
        float b = -INFINITY, p = -INFINITY;
        int   a = 0;
        for (int w = 0; w < nwarp; ++w) {
            if (shb[u*nwarp + w] > b || (shb[u*nwarp + w] == b && sha[u*nwarp + w] < a)) {
                b = shb[u*nwarp + w];
                a = sha[u*nwarp + w];
            }
            p = fmaxf(p, shp[u*nwarp + w]);
        }
        const size_t o = (size_t) h*M + m0 + u;
        best_score[o] = b;
        best_slot [o] = a == INT_MAX ? 0 : a;
        sel       [o] = fmaxf(b, p);
    }
}

// 2. the n_append least familiar columns become slots.  One block per head does
//    a bitonic sort of (sel, index) so the split matches the CPU exactly.
static __global__ void lod2_k_split(
        const float * __restrict__ sel, const int M, const int n_append,
        int * __restrict__ is_app, int * __restrict__ app_list, int * __restrict__ owner,
        const int s_len) {
    const int h = blockIdx.x;

    extern __shared__ char raw[];
    float * key = (float *) raw;
    int   * idx = (int *) (key + blockDim.x*2);

    const int N = blockDim.x*2; // padded power of two, >= M
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        key[i] = i < M ? sel[(size_t) h*M + i] : INFINITY;
        idx[i] = i;
    }
    __syncthreads();

    for (int len = 2; len <= N; len <<= 1) {
        for (int stride = len >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < N/2; i += blockDim.x) {
                const int lo = 2*i - (i & (stride - 1));
                const int hi = lo + stride;
                const bool up = ((lo & len) == 0);
                const bool swap = (key[lo] > key[hi]) ||
                                  (key[lo] == key[hi] && idx[lo] > idx[hi]);
                if (swap == up) {
                    const float tk = key[lo]; key[lo] = key[hi]; key[hi] = tk;
                    const int   ti = idx[lo]; idx[lo] = idx[hi]; idx[hi] = ti;
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        is_app[(size_t) h*M + i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_append; i += blockDim.x) {
        is_app[(size_t) h*M + idx[i]] = 1;
    }
    __syncthreads();

    // appended columns take slots in ascending column order
    if (threadIdx.x == 0) {
        int j = 0;
        for (int i = 0; i < M; ++i) {
            if (is_app[(size_t) h*M + i]) {
                app_list[(size_t) h*M + j] = i;
                owner   [(size_t) h*M + i] = s_len + j;
                j++;
            }
        }
    }
}

static __global__ void lod2_k_iota(int * __restrict__ owner, const int M) {
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        owner[(size_t) blockIdx.x*M + m] = m;
    }
}

// 3. write the appended slots, and seed the state when it is still empty
static __global__ void lod2_k_append(
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        float * __restrict__ s_kv, const int S_cap, const int Ds, const int D, const int Dv,
        const int p0, const int M, const int s_len,
        const int * __restrict__ app_list, const bool seed) {
    const int j = blockIdx.x;
    const int h = blockIdx.y;

    const int m = seed ? j : app_list[(size_t) h*M + j];
    float * row = s_kv + ((size_t) h*S_cap + s_len + j)*Ds;

    const char * krow = k + (size_t)(p0 + m)*knb1 + (size_t) h*knb2;
    const char * vrow = v + (size_t)(p0 + m)*vnb1 + (size_t) h*vnb2;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row[i] = lod2_get(krow, ktype, i);
    }
    for (int i = threadIdx.x; i < Dv; i += blockDim.x) {
        row[D + i] = lod2_get(vrow, vtype, i);
    }
    if (threadIdx.x == 0) {
        row[D + Dv] = 1.0f;
    }
}

// 4. merge destinations: the best of the old slots and the columns just appended
static __global__ void lod2_k_dest(
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const int D, const int p0, const int M, const int s_len, const int n_append,
        const float * __restrict__ best_score, const int * __restrict__ best_slot,
        const int * __restrict__ is_app, const int * __restrict__ app_list,
        int * __restrict__ owner) {
    const int m = blockIdx.x;
    const int h = blockIdx.y;
    const size_t o = (size_t) h*M + m;

    if (is_app[o]) {
        return;
    }

    extern __shared__ float smem[];
    float * kq = smem;

    const char * krow = k + (size_t)(p0 + m)*knb1 + (size_t) h*knb2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        kq[i] = lod2_get(krow, ktype, i);
    }
    __syncthreads();

    const int lane  = threadIdx.x % LOD2_LANES;
    const int warp  = threadIdx.x / LOD2_LANES;
    const int nwarp = blockDim.x / LOD2_LANES;

    float wbest = best_score[o];
    int   warg  = -1;

    for (int j = warp; j < n_append; j += nwarp) {
        const int ma = app_list[(size_t) h*M + j];
        const char * arow = k + (size_t)(p0 + ma)*knb1 + (size_t) h*knb2;
        float acc = 0.0f;
        for (int i = lane; i < D; i += LOD2_LANES) {
            acc += kq[i]*lod2_get(arow, ktype, i);
        }
        acc = lod2_warp_sum(acc);
        // strictly greater: the first appended slot wins ties, as on the CPU
        if (acc > wbest) {
            wbest = acc;
            warg  = j;
        }
    }

    __shared__ float sh_best[LOD2_BLOCK/2];
    __shared__ int   sh_arg [LOD2_BLOCK/2];
    if (lane == 0) {
        sh_best[warp] = wbest;
        sh_arg [warp] = warg;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float b = best_score[o];
        int   a = -1;
        for (int w = 0; w < nwarp; ++w) {
            if (sh_arg[w] >= 0 && (sh_best[w] > b || (sh_best[w] == b && (a < 0 || sh_arg[w] < a)))) {
                b = sh_best[w];
                a = sh_arg[w];
            }
        }
        owner[o] = a < 0 ? best_slot[o] : s_len + a;
    }
}

// 5. apply the merges.  One block per slot scanning columns in order keeps the
//    accumulation deterministic without atomics.
static __global__ void lod2_k_merge(
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        float * __restrict__ s_kv, const int S_cap, const int Ds, const int D, const int Dv,
        const int p0, const int M,
        const int * __restrict__ is_app, const int * __restrict__ owner) {
    const int s = blockIdx.x;
    const int h = blockIdx.y;

    float * row = s_kv + ((size_t) h*S_cap + s)*Ds;

    float add = 0.0f;
    for (int m = 0; m < M; ++m) {
        const size_t o = (size_t) h*M + m;
        if (is_app[o] || owner[o] != s) {
            continue;
        }
        const char * krow = k + (size_t)(p0 + m)*knb1 + (size_t) h*knb2;
        const char * vrow = v + (size_t)(p0 + m)*vnb1 + (size_t) h*vnb2;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            row[i] += lod2_get(krow, ktype, i);
        }
        for (int i = threadIdx.x; i < Dv; i += blockDim.x) {
            row[D + i] += lod2_get(vrow, vtype, i);
        }
        add += 1.0f;
    }
    if (threadIdx.x == 0 && add > 0.0f) {
        row[D + Dv] += add;
    }
}

// 7. publish the means.  Routing and the coarse branch read these every token
// and the sums only when a route is opened, so keeping a maintained F16 copy
// halves that traffic and removes a divide from the inner loop.
static __global__ void lod2_k_means(
        const float * __restrict__ s_kv, float * __restrict__ s_mn,
        const int S_cap, const int Ds, const int D, const int Dv) {
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const float * row = s_kv + ((size_t) h*S_cap + s)*Ds;
    float       * out = s_mn + ((size_t) h*S_cap + s)*(Ds - 1);
    const float cnt = row[D + Dv];
    const float inv = cnt > 0.0f ? 1.0f/cnt : 0.0f;
    for (int i = threadIdx.x; i < Ds - 1; i += blockDim.x) {
        out[i] = row[i]*inv;
    }
}

// 6. page append: one block per slot, columns in arrival order
static __global__ void lod2_k_pages(
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        float * __restrict__ p_kv, int * __restrict__ p_idx, int * __restrict__ s_pg,
        int * __restrict__ meta,
        const int S_cap, const int P_cap, const int Ds, const int D, const int Dv,
        const int ps, const int pps, const int p0, const int M,
        const int * __restrict__ owner) {
    const int s = blockIdx.x;
    const int h = blockIdx.y;

    int * ent = s_pg + ((size_t) h*S_cap + s)*(pps + 1);

    __shared__ int sh_len;
    __shared__ int sh_base;

    // how many of this block's columns land in this slot
    int n_own = 0;
    for (int m = 0; m < M; ++m) {
        n_own += (owner[(size_t) h*M + m] == s);
    }
    if (n_own == 0) {
        return;
    }

    if (threadIdx.x == 0) {
        const int len = ent[pps];
        const int old_pages = (len + ps - 1)/ps;
        const int new_pages = (len + n_own + ps - 1)/ps - old_pages;
        sh_len  = len;
        sh_base = new_pages > 0 ? atomicAdd(&meta[(size_t) h*4], new_pages) : 0;
        for (int t = 0; t < new_pages; ++t) {
            ent[old_pages + t] = sh_base + t;
        }
        ent[pps] = len + n_own;
    }
    __syncthreads();

    int r = 0;
    for (int m = 0; m < M; ++m) {
        if (owner[(size_t) h*M + m] != s) {
            continue;
        }
        const int index = sh_len + r;
        const int ord   = index/ps;
        const int lane  = index%ps;
        const int page  = ent[ord];

        float * prow = p_kv + ((size_t) h*P_cap + page)*Ds;
        const char * krow = k + (size_t)(p0 + m)*knb1 + (size_t) h*knb2;
        const char * vrow = v + (size_t)(p0 + m)*vnb1 + (size_t) h*vnb2;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            prow[i] += lod2_get(krow, ktype, i);
        }
        for (int i = threadIdx.x; i < Dv; i += blockDim.x) {
            prow[D + i] += lod2_get(vrow, vtype, i);
        }
        if (threadIdx.x == 0) {
            prow[D + Dv] += 1.0f;
            p_idx[((size_t) h*P_cap + page)*ps + lane] = p0 + m;
        }
        r++;
    }
}

// ---------------------------------------------------------------------------
// attention: one warp per (token, query head)

struct lod2_acc {
    float m;   // running max, base two
    float d;   // running denominator
};

static __device__ __forceinline__ void lod2_acc_init(lod2_acc & a, float * acc, const int n) {
    a.m = -INFINITY;
    a.d = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc[i] = 0.0f;
    }
}

static __global__ __launch_bounds__(LOD2_BLOCK) void lod2_k_attn(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        const float * __restrict__ s_kv, const float * __restrict__ s_mn,
        const float * __restrict__ lgt,
        const float * __restrict__ p_kv,
        const int * __restrict__ p_idx, const int * __restrict__ s_pg,
        char * __restrict__ dst, const size_t dnb1, const size_t dnb2,
        const int S_cap, const int P_cap, const int Ds, const int D, const int Dv,
        const int ps, const int pps, const int g,
        const int l0, const int * __restrict__ q0p, const int nq, const int s_len,
        const int lstride, const int n_routes, const int sink, const float scale,
        const float * __restrict__ lm_in, const float * __restrict__ ld_in,
        const float * __restrict__ lacc_in,
        const float * __restrict__ cm_in, const float * __restrict__ cd_in,
        const float * __restrict__ cacc_in) {
    const int q0 = q0p[0];
    // One warp per (token, query head).  Staging the slot rows in shared memory
    // was tried here - the four warps of a block do walk the same slots - and it
    // measured *slower* on both models (see docs/lod2-port-handover.md): the
    // whole state fits in the 256 MB infinity cache, so the "redundant" reads
    // were already cache hits, and the syncs needed to stage them are not free.
    const int t    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int hq   = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const bool active = t < nq;   // no early return: the staging syncs are collective
    const int h = hq/g;


    const float sl2 = scale*LOD2_LOG2E;

    const float * slot = s_kv + (size_t) h*S_cap*Ds;
    const float * page = p_kv + (size_t) h*P_cap*Ds;
    const int   * pidx = p_idx + (size_t) h*P_cap*ps;
    const int   * spg  = s_pg  + (size_t) h*S_cap*(pps + 1);

    const float * qv = (const float *)(q + (size_t)(active ? t : 0)*qnb1 + (size_t) hq*qnb2);
    const float * mn  = s_mn + (size_t) h*S_cap*(Ds - 1);
    // q . mean_k for every slot, produced by ggml_mul_mat: the author's Triton
    // path does exactly this and hands the contraction to the BLAS GEMM
    const float * lg  = lgt + (size_t)((size_t) hq*nq + (active ? t : 0))*lstride;

    // --- 5.1 routing ------------------------------------------------------
    const int protect = min(sink, s_len);
    const int R       = min(n_routes, max(s_len - protect, 0));

    int   routes[LOD2_MAX_R];
    float rscore[LOD2_MAX_R];
    int   n_open = 0;
#pragma unroll
    for (int r = 0; r < LOD2_MAX_R; ++r) {
        routes[r] = -1;
        rscore[r] = -INFINITY;
    }

    for (int s0 = protect; s0 < s_len; s0 += LOD2_SB) {
        const int nb = min(LOD2_SB, s_len - s0);
        if (!active) {
            continue;
        }
        for (int j = 0; j < nb; ++j) {
        const int s = s0 + j;
        const float cnt = slot[(size_t) s*Ds + D + Dv];
        if (!(cnt > 0.5f)) {
            continue;
        }
        const float lgv = sl2*lg[s] + log2f(cnt);

        // insertion into the (small, descending) route list
        if (n_open < R || lgv > rscore[R - 1]) {
            int pos = min(n_open, R - 1);
            while (pos > 0 && rscore[pos - 1] < lgv) {
                rscore[pos] = rscore[pos - 1];
                routes[pos] = routes[pos - 1];
                pos--;
            }
            rscore[pos] = lgv;
            routes[pos] = s;
            n_open = min(n_open + 1, R);
        }
        }
    }

    // --- 5.2 coarse branch -------------------------------------------------
    // The opened slots are only known after the pass above, so the logits are
    // recomputed here rather than subtracted out: they are the largest ones,
    // and removing them from a finished denominator loses precision exactly
    // when the routed branch matters most.
    float acc_c[16];
    lod2_acc c;
    const int epl = (Dv + WARP_SIZE - 1)/WARP_SIZE;
    lod2_acc_init(c, acc_c, epl);

    // Nothing below here is collective, so the inactive warps can leave: the
    // "no early return" note above dates from a staging step that is gone.
    if (!active) {
        return;
    }

    if (cm_in) {
        // lod2_k_coarse already did this branch, tiled
        const size_t unit = (size_t) hq*nq + t;
        c.m = cm_in[unit];
        c.d = cd_in[unit];
        for (int e = 0; e < epl; ++e) {
            const int i = lane + e*WARP_SIZE;
            acc_c[e] = i < Dv ? cacc_in[unit*Dv + i] : 0.0f;
        }
    } else
    for (int s0 = 0; s0 < s_len; s0 += LOD2_SB) {
        const int nb = min(LOD2_SB, s_len - s0);
        if (!active) {
            continue;
        }
        for (int j = 0; j < nb; ++j) {
        const int s = s0 + j;
        bool opened = false;
        for (int r = 0; r < n_open; ++r) {
            opened |= (routes[r] == s);
        }
        if (opened) {
            continue;
        }
        const float cnt = slot[(size_t) s*Ds + D + Dv];
        if (!(cnt > 0.5f)) {
            continue;
        }
        const float lgv = sl2*lg[s] + log2f(cnt);

        const float nm = fmaxf(c.m, lgv);
        const float co = c.m == -INFINITY ? 0.0f : exp2f(c.m - nm);
        const float pr = exp2f(lgv - nm);
        c.d = c.d*co + pr;
        for (int e = 0; e < epl; ++e) {
            const int i = lane + e*WARP_SIZE;
            acc_c[e] = acc_c[e]*co + (i < Dv ? pr*mn[(size_t) s*(Ds - 1) + D + i] : 0.0f);
        }
        c.m = nm;
        }
    }

    // --- 5.3 local branch --------------------------------------------------
    float acc_l[16];
    lod2_acc l;
    lod2_acc_init(l, acc_l, epl);

    // Column tile, after the author's Triton local loop: score the tile, take
    // its maximum, apply ONE correction to the accumulator, then accumulate the
    // tile's values.  Doing it a column at a time - which is what this was -
    // puts an exp2f between every score and the next, so the loop runs at the
    // latency of a dependent transcendental chain and the warp reductions
    // cannot overlap either.  Scoring LOD2_QT columns first makes those
    // reductions independent and pays one rescale per tile instead of per
    // column.  Same transformation that was worth 4.1x on the decode kernel.
    if (lm_in) {
        // lod2_k_local already did this branch, tiled
        const size_t unit = (size_t) hq*nq + t;
        l.m = lm_in[unit];
        l.d = ld_in[unit];
        for (int e = 0; e < epl; ++e) {
            const int i = lane + e*WARP_SIZE;
            acc_l[e] = i < Dv ? lacc_in[unit*Dv + i] : 0.0f;
        }
    } else
    for (int j0 = l0; j0 <= q0 + t; j0 += LOD2_QT) {
        const int n = min(LOD2_QT, q0 + t - j0 + 1);

        float sc[LOD2_QT];
#pragma unroll
        for (int u = 0; u < LOD2_QT; ++u) {
            float a = 0.0f;
            if (u < n) {
                const char * krow = k + (size_t)(j0 + u)*knb1 + (size_t) h*knb2;
                for (int i = lane; i < D; i += WARP_SIZE) {
                    a += qv[i]*lod2_get(krow, ktype, i);
                }
            }
            sc[u] = a;
        }
#pragma unroll
        for (int u = 0; u < LOD2_QT; ++u) {
            sc[u] = lod2_wsum<WARP_SIZE>(sc[u])*sl2;
        }

        float tm = -INFINITY;
#pragma unroll
        for (int u = 0; u < LOD2_QT; ++u) {
            if (u < n) {
                tm = fmaxf(tm, sc[u]);
            }
        }

        const float nm = fmaxf(l.m, tm);
        const float co = l.m == -INFINITY ? 0.0f : exp2f(l.m - nm);
        l.d *= co;
        for (int e = 0; e < epl; ++e) {
            acc_l[e] *= co;
        }
        l.m = nm;

        for (int u = 0; u < n; ++u) {
            const char * vrow = v + (size_t)(j0 + u)*vnb1 + (size_t) h*vnb2;
            const float pr = exp2f(sc[u] - nm);
            l.d += pr;
            for (int e = 0; e < epl; ++e) {
                const int i = lane + e*WARP_SIZE;
                acc_l[e] += i < Dv ? pr*lod2_get(vrow, vtype, i) : 0.0f;
            }
        }
    }

    // --- 5.4 routed branch -------------------------------------------------
    float acc_e[16];
    lod2_acc e;
    lod2_acc_init(e, acc_e, epl);

    for (int r = 0; r < n_open; ++r) {
        const int s = routes[r];
        const int * ent = spg + (size_t) s*(pps + 1);
        const int n_pg = (ent[pps] + ps - 1)/ps;

        int   sel  = -1;
        float best = -INFINITY;
        for (int o = 0; o < n_pg; ++o) {
            const int p = ent[o];
            if (p < 0) {
                continue;
            }
            const float * prow = page + (size_t) p*Ds;
            const float cnt = prow[D + Dv];
            if (!(cnt > 0.0f)) {
                continue;
            }
            float a = 0.0f;
            for (int i = lane; i < D; i += WARP_SIZE) {
                a += qv[i]*prow[i];
            }
            a = lod2_wsum<WARP_SIZE>(a);
            const float sc = sl2*(a/cnt) + log2f(cnt);
            if (sc > best) {
                best = sc;
                sel  = p;
            }
        }
        if (sel < 0) {
            continue;
        }

        const float * prow = page + (size_t) sel*Ds;
        const float * srow = slot + (size_t) s*Ds;
        const float p_cnt = prow[D + Dv];
        const float r_cnt = srow[D + Dv] - p_cnt;

        if (r_cnt > 0.0f) {
            const float inv = 1.0f/r_cnt;
            float a = 0.0f;
            for (int i = lane; i < D; i += WARP_SIZE) {
                a += qv[i]*(srow[i] - prow[i]);
            }
            a = lod2_wsum<WARP_SIZE>(a);
            const float lg = sl2*(a*inv) + log2f(r_cnt);

            const float nm = fmaxf(e.m, lg);
            const float co = e.m == -INFINITY ? 0.0f : exp2f(e.m - nm);
            const float pr = exp2f(lg - nm);
            e.d = e.d*co + pr;
            for (int i = 0; i < epl; ++i) {
                const int d = lane + i*WARP_SIZE;
                acc_e[i] = acc_e[i]*co + (d < Dv ? pr*(srow[D + d] - prow[D + d])*inv : 0.0f);
            }
            e.m = nm;
        }

        const int n_tok = min((int) p_cnt, ps);
        for (int i = 0; i < n_tok; ++i) {
            const int col = pidx[(size_t) sel*ps + i];
            if (col < 0) {
                continue;
            }
            const char * krow = k + (size_t) col*knb1 + (size_t) h*knb2;
            const char * vrow = v + (size_t) col*vnb1 + (size_t) h*vnb2;
            float a = 0.0f;
            for (int d = lane; d < D; d += WARP_SIZE) {
                a += qv[d]*lod2_get(krow, ktype, d);
            }
            a = lod2_wsum<WARP_SIZE>(a)*sl2;

            const float nm = fmaxf(e.m, a);
            const float co = e.m == -INFINITY ? 0.0f : exp2f(e.m - nm);
            const float pr = exp2f(a - nm);
            e.d = e.d*co + pr;
            for (int x = 0; x < epl; ++x) {
                const int d = lane + x*WARP_SIZE;
                acc_e[x] = acc_e[x]*co + (d < Dv ? pr*lod2_get(vrow, vtype, d) : 0.0f);
            }
            e.m = nm;
        }
    }

    // --- 5.5 merge ---------------------------------------------------------
    const float lc = c.d > 0.0f ? c.m + log2f(c.d) : -INFINITY;
    const float ll = l.d > 0.0f ? l.m + log2f(l.d) : -INFINITY;
    const float le = e.d > 0.0f ? e.m + log2f(e.d) : -INFINITY;
    const float mx = fmaxf(lc, fmaxf(ll, le));

    const float wc = lc > -INFINITY ? exp2f(lc - mx) : 0.0f;
    const float wl = ll > -INFINITY ? exp2f(ll - mx) : 0.0f;
    const float we = le > -INFINITY ? exp2f(le - mx) : 0.0f;
    const float inv = 1.0f/(wc + wl + we);

    float * out = (float *)(dst + (size_t) hq*dnb1 + (size_t) t*dnb2);
    for (int x = 0; x < epl; ++x) {
        const int d = lane + x*WARP_SIZE;
        if (d < Dv) {
            const float o =
                (c.d > 0.0f ? wc*acc_c[x]/c.d : 0.0f) +
                (l.d > 0.0f ? wl*acc_l[x]/l.d : 0.0f) +
                (e.d > 0.0f ? we*acc_e[x]/e.d : 0.0f);
            out[d] = o*inv;
        }
    }
}

// ---------------------------------------------------------------------------
// attention, split decode path
//
// One warp per (token, query head) is fine while a ubatch supplies thousands of
// units, and useless during generation: a single token launches n_head_q warps.
// So for short ubatches the work is cut differently.
//
// The column space of one unit is the concatenation
//     [ s_len coarse slots | local columns | routed entries ]
// and an online softmax over a partition merges exactly, so any split works.
// The routing has to see all the slots at once, which is why the logits are
// computed once into scratch and then reused: the slot keys are read once, the
// slot values once, and nothing is recomputed.

#define LOD2_DEC_MAX_Q 8   // ubatches this wide or narrower take the split path
#define LOD2_DEC_MAX_S 64  // splits per unit

// one thread per (unit, slot): apply the count bias to the GEMM output and lay
// it out unit-major for the split kernels
static __global__ void lod2_k_bias(
        const float * __restrict__ s_kv, const float * __restrict__ lgt,
        float * __restrict__ logits,
        const int S_cap, const int Ds, const int D, const int Dv,
        const int nq, const int Hq, const int g, const int s_len, const int lstride,
        const float sl2) {
    const int w = blockIdx.x*blockDim.x + threadIdx.x;
    const int s    = w % s_len;
    const int unit = w / s_len;
    if (unit >= nq*Hq) {
        return;
    }
    const int t  = unit / Hq;
    const int hq = unit % Hq;
    const int h  = hq / g;
    const float cnt = s_kv[((size_t) h*S_cap + s)*Ds + D + Dv];
    logits[(size_t) unit*s_len + s] = cnt > 0.5f
        ? sl2*lgt[(size_t)((size_t) hq*nq + t)*lstride + s] + log2f(cnt)
        : -INFINITY;
}

// ---------------------------------------------------------------------------
// tiled local branch (opt-in, LLAMA_LOD2_TILED)
//
// A port of the author's Triton local loop rather than of its arithmetic: a
// block owns LOD2_TM queries of one query head and walks the window in LOD2_TN
// column tiles, staging Q, K and V in LDS.  Two things follow, and they are the
// two that section 4d.1 identified as the only ones that can move prefill:
//
//   - a K/V column is read from memory once per LOD2_TM queries, not once per
//     query;
//   - a score is one thread's serial dot over LDS, so the six-step cross-lane
//     reduction that the per-query kernel pays for *every* (query, column) pair
//     disappears.  What is left of the softmax is one tile maximum and one
//     correction per tile, as in `tl.dot` + `tl.max(scores, axis=1)`.
//
// Thread t owns query m = t/LOD2_TN throughout, in both the score mapping
// (n = t%LOD2_TN) and the value mapping (dim = t%LOD2_TN + LOD2_TN*e), so the
// per-query softmax state stays in registers replicated across the LOD2_TN
// threads of a row - no LDS and no barrier for it.
//
// Q, K and V are staged transposed where the access pattern wants it: ks is
// [D][TN] so the TN threads of a score row read consecutive floats, and vs is
// [TN][Dv] so the TN threads of a value row do the same.
#define LOD2_TM 16
#define LOD2_TN 16
#define LOD2_TACC 16   // Dv/LOD2_TN, bounded by the Dv <= 256 launch guard


// KVH stages the K and value tiles in shared memory as half instead of float.
// K and V already live as F16 in the cache, so lod2_get's F16 -> F32 widening
// is undone exactly by narrowing back: every float that reaches the arithmetic
// is bit-identical to the F32-staged path.  What it buys is the two largest
// buffers halved, which is what takes this kernel from one workgroup per CU to
// two, and halves its shared-memory read traffic on the way.
template <int TM, bool KVH>
static __global__ __launch_bounds__(TM*LOD2_TN) void lod2_k_local(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        float * __restrict__ lm, float * __restrict__ ld, float * __restrict__ lacc,
        const int D, const int Dv, const int g,
        const int l0, const int * __restrict__ q0p, const int nq, const int Hq,
        const float sl2) {
    const int q0  = q0p[0];
    const int hq  = blockIdx.y;
    const int h   = hq/g;
    const int t0  = blockIdx.x*TM;
    const int tid = threadIdx.x;
    const int m   = tid/LOD2_TN;
    const int n   = tid%LOD2_TN;

    const int tq  = t0 + m;                 // this thread's query
    const bool qok = tq < nq;
    const int  qend = q0 + tq;              // last column this query may see

    // Layout: Q tile, then the K tile, then the value tile.  ps is not live at
    // the same time as the K tile - the scores are finished with it before any
    // probability is written - so it is folded into the K tile's bytes, which
    // is the last 1024 that stands between this kernel and two workgroups per
    // CU.  The extra barrier that aliasing needs is the one marked below.
    extern __shared__ char lsmb[];
    const size_t kbytes = KVH ? (size_t) D*LOD2_TN*sizeof(half) : (size_t) D*LOD2_TN*sizeof(float);
    const size_t kpb    = kbytes > (size_t) TM*LOD2_TN*sizeof(float)
                        ? kbytes : (size_t) TM*LOD2_TN*sizeof(float);

    // Q and K are held as adjacent pairs along D - [D/2][TM] of float2 and
    // [D/2][TN] of float2/half2 - so the score loop takes two elements per LDS
    // instruction and the two multiplies it then does are independent and land
    // in adjacent registers, which is what v_pk_fma_f32 needs.
    const int D2 = D/2;

    float2 * qs = (float2 *) lsmb;                      // [D/2][TM]
    char   * ksb = lsmb + (size_t) D*TM*sizeof(float);  // [D/2][TN]
    float  * ps = (float *) ksb;                        // [TM][TN], aliased
    char   * vsb = ksb + kpb;                           // [TN][Dv], V or half V

    float2 * ksf = (float2 *) ksb;
    half2  * ksh = (half2  *) ksb;
    float  * vsf = (float  *) vsb;
    half   * vsh = (half   *) vsb;

    // stage Q once for the whole window
    for (int i = tid; i < D2*TM; i += TM*LOD2_TN) {
        const int d2 = i/TM;
        const int mm = i%TM;
        const int tt = t0 + mm;
        const float * qr = (const float *)(q + (size_t) tt*qnb1 + (size_t) hq*qnb2);
        qs[i] = tt < nq ? make_float2(qr[2*d2], qr[2*d2 + 1]) : make_float2(0.0f, 0.0f);
    }

    float acc[LOD2_TACC];
#pragma unroll
    for (int e = 0; e < LOD2_TACC; ++e) {
        acc[e] = 0.0f;
    }
    float m_run = -INFINITY;
    float d_run = 0.0f;

    // the block's queries end at different columns; run to the latest
    const int jend = q0 + min(t0 + TM - 1, nq - 1);
    const int nacc = Dv/LOD2_TN;

    for (int j0 = l0; j0 <= jend; j0 += LOD2_TN) {
        __syncthreads();
        for (int i = tid; i < D2*LOD2_TN; i += TM*LOD2_TN) {
            const int d2 = i/LOD2_TN;
            const int nn = i%LOD2_TN;
            const int c  = j0 + nn;
            const char * kr = k + (size_t) c*knb1 + (size_t) h*knb2;
            const float x0 = c <= jend ? lod2_get(kr, ktype, 2*d2)     : 0.0f;
            const float x1 = c <= jend ? lod2_get(kr, ktype, 2*d2 + 1) : 0.0f;
            if (KVH) { ksh[i] = __floats2half2_rn(x0, x1); } else { ksf[i] = make_float2(x0, x1); }
        }
        for (int i = tid; i < LOD2_TN*Dv; i += TM*LOD2_TN) {
            const int nn = i/Dv;
            const int d  = i%Dv;
            const int c  = j0 + nn;
            const float x = c <= jend ? lod2_get(v + (size_t) c*vnb1 + (size_t) h*vnb2, vtype, d) : 0.0f;
            if (KVH) { vsh[i] = __float2half(x); } else { vsf[i] = x; }
        }
        __syncthreads();

        // score: one thread, one (query, column), serial over D in LDS.
        // Tried and reverted: [TM][DP] / [TN][DP] rows read four floats at a
        // time.  That cuts the LDS instruction count by four and makes the
        // staging loads coalesced, and it measured 13.9% SLOWER (1086.5 ->
        // 936.0 t/s).  This loop is not issue bound, so trading bank behaviour
        // for instruction count loses.  See docs/lod2-port-handover.md 4d.3.
        // Tried and reverted: four independent partial sums, to break what
        // looks like a serial dependent FMA chain D long.  No effect (1088.1
        // -> 1082.0 t/s), so the compiler is already scheduling across it and
        // the stall is the LDS traffic itself, not the accumulator.
        float2 a = make_float2(0.0f, 0.0f);
        for (int d2 = 0; d2 < D2; ++d2) {
            const float2 qq = qs[(size_t) d2*TM + m];
            const float2 kk = KVH ? __half22float2(ksh[(size_t) d2*LOD2_TN + n])
                                  : ksf[(size_t) d2*LOD2_TN + n];
            a.x += qq.x*kk.x;
            a.y += qq.y*kk.y;
        }
        float s = a.x + a.y;
        const int c = j0 + n;
        const bool ok = qok && c <= qend;
        s = ok ? s*sl2 : -INFINITY;

        // tile maximum and denominator over the row: LOD2_TN consecutive lanes
        float tm = s;
#pragma unroll
        for (int off = LOD2_TN/2; off > 0; off >>= 1) {
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffff, tm, off, LOD2_TN));
        }

        const float nm = fmaxf(m_run, tm);
        const float co = m_run == -INFINITY ? 0.0f : exp2f(m_run - nm);
        const float pr = nm == -INFINITY ? 0.0f : exp2f(s - nm);

        float rs = pr;
#pragma unroll
        for (int off = LOD2_TN/2; off > 0; off >>= 1) {
            rs += __shfl_xor_sync(0xffffffff, rs, off, LOD2_TN);
        }
        d_run = d_run*co + rs;
        m_run = nm;

        // ps aliases the K tile: every thread must be done reading K first
        __syncthreads();
        ps[m*LOD2_TN + n] = pr;
        __syncthreads();

        // value: same m, this thread owns every LOD2_TN-th dimension.
        // Tried and reverted: column-outer / dimension-inner, which loads
        // ps[m][u] once per column instead of once per (column, dimension).
        // Bit-identical and it measured slower on both kernels.
        for (int e = 0; e < nacc; ++e) {
            const int dv = n + e*LOD2_TN;
            float a = acc[e]*co;
            for (int u = 0; u < LOD2_TN; ++u) {
                const float vv = KVH ? __half2float(vsh[(size_t) u*Dv + dv])
                                     : vsf[(size_t) u*Dv + dv];
                a += ps[m*LOD2_TN + u]*vv;
            }
            acc[e] = a;
        }
    }

    if (!qok) {
        return;
    }
    const size_t unit = (size_t) hq*nq + tq;
    if (n == 0) {
        lm[unit] = m_run;
        ld[unit] = d_run;
    }
    for (int e = 0; e < nacc; ++e) {
        lacc[unit*Dv + n + e*LOD2_TN] = acc[e];
    }
}

// Local branch, dimension split.  The kernel above gives a thread one (query,
// column) pair, so its dot product reads D floats of Q and D of K out of shared
// memory to produce 2D flops - one element moved per flop, no reuse inside the
// thread at all - and the value pass is worse still, because it reads the
// probability back out of shared memory once per (column, dimension).  Counted
// per thread per column tile that is 768 shared-memory instructions for 1024
// flops, which is why rocprof puts this branch at ~3.7 TFLOPS against 72.7 for
// llama.cpp's own flash attention on the same device, and why every attempt to
// fix it from the instruction mix (float4 rows, split accumulators, packed f32)
// moved it by at most 1.3%.  The ceiling is the shared-memory pipe.
//
// Here the LOD2_TN lanes of a row split the head dimension instead of the
// columns: lane n owns dims [n*DP, (n+1)*DP) of query m for the whole window.
// That changes three things at once.
//   - Q lives in registers, loaded once from global, never re-read.
//   - A lane reads DP contiguous elements of K per column, so the compiler
//     issues wide ds_read instead of DP scalar ones.
//   - The probabilities stay in registers, so the ps round trip through shared
//     memory and the two barriers around it disappear entirely.
// The same tile then costs 64 shared-memory instructions instead of 768, at the
// price of one cross-lane reduction per column.  The staged tile is [LOD2_TN][D]
// and [LOD2_TN][Dv], so the footprint no longer depends on TM and the row count
// is free to grow.
//
// The dot product is summed per lane and then across lanes where the kernel
// above sums it serially in one lane, so the two are not bit-identical.  The
// merge that consumes this is a logsumexp over independent branches and does
// not care; correctness is checked the usual way, against dense with every slot
// open.
//
// DP is a template parameter because the register arrays have to be indexed by
// compile-time constants - a runtime bound would spill them to scratch and lose
// the entire point.  D == Dv is a launch precondition for the same reason.
// RM is how many queries one thread carries.  The dimension split alone still
// has all LOD2_TN*RG threads that share a K element convert it separately: the
// disassembly of the RM == 1 form is 512 v_cvt_f32_f16 per column tile against
// 264 v_pk_fma_f32 that do the arithmetic, so more than half the instructions
// are widening the same values over and over.  Carrying RM queries per thread
// converts each element once and spends it RM times, and amortises the address
// arithmetic and the s_waitcnt chain with it.  The tile itself does not grow -
// only the register file does, by RM*(2*DP + LOD2_TN) floats.
// LOD2_CH is how many head dimensions a lane takes in one run before the next
// lane's run starts.  Contiguous slices - lane n owning [n*DP, (n+1)*DP) - look
// natural and are the reason this kernel was stuck: with DP 16 halves a lane's
// slice starts every 32 bytes, so the sixteen lanes of a row land on eight banks
// and every K and V read is a four-way conflict.  Striping in runs of four
// instead puts lane n at dword 2n within a run, so the sixteen lanes cover
// thirty-two consecutive dwords - one per bank - at the same eight bytes per
// lane and the same instruction count.
//
// This is what the f32 tile experiment proved.  Staging f32 removes all 512
// conversions per tile and measured 1.70x SLOWER at identical shared-memory
// footprint, occupancy and instruction count, because f32 doubles the conflict.
// The kernel is bound by the shared-memory pipe, not by issue - which is also
// why carrying two queries per thread, which cuts instructions by a third, was
// worth 0.3%.
#define LOD2_CH 4
#define LOD2_DSD(c) ((c)*LOD2_CH*LOD2_TN + n*LOD2_CH)

// CT is the column tile.  It is a free parameter here, unlike in the kernels
// above: the lanes of a row carry the head dimension, not columns, so nothing
// ties the staged column count to LOD2_TN.  That is what makes the f32 tile
// affordable.  Staging K and V as f32 removes every conversion in the kernel -
// lod2_get already returns f32, so the narrowing on the store side goes too -
// which is 512 of the 1990 instructions per tile.  It doubles the tile, so CT
// is halved to keep the shared-memory footprint and therefore the occupancy
// exactly where it was.
template <int RG, int RM, int CT, int DP, bool KVH>
static __global__ __launch_bounds__(RG*LOD2_TN) void lod2_k_local_ds(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        float * __restrict__ lm, float * __restrict__ ld, float * __restrict__ lacc,
        const int D, const int Dv, const int g,
        const int l0, const int * __restrict__ q0p, const int nq, const int Hq,
        const float sl2) {
    const int TM  = RG*RM;                  // queries per block
    const int q0  = q0p[0];
    const int hq  = blockIdx.y;
    const int h   = hq/g;
    const int t0  = blockIdx.x*TM;
    const int tid = threadIdx.x;
    const int m   = tid/LOD2_TN;
    const int n   = tid%LOD2_TN;

    int  tq  [RM];
    bool qok [RM];
    int  qend[RM];
#pragma unroll
    for (int r = 0; r < RM; ++r) {
        tq  [r] = t0 + m*RM + r;
        qok [r] = tq[r] < nq;
        qend[r] = q0 + tq[r];
    }

    // The tile stride is written as DP*LOD2_TN rather than D even though the two
    // are equal by construction: D is a kernel argument, so with D the compiler
    // cannot see that a lane's slice starts on a DP boundary and falls back to
    // one ds_read per element.  Spelled as a compile-time constant it proves the
    // alignment and the DP reads collapse into two ds_read_b128.
    const int DPT = DP*LOD2_TN;

    extern __shared__ char lsmb[];
    const size_t kvw = KVH ? sizeof(half) : sizeof(float);
    char * ksb = lsmb;                              // [CT][D]
    char * vsb = lsmb + (size_t) CT*DPT*kvw;        // [CT][Dv]

    const half  * ksh = (const half  *) ksb;
    const float * ksf = (const float *) ksb;
    const half  * vsh = (const half  *) vsb;
    const float * vsf = (const float *) vsb;

    half  * kshw = (half  *) ksb;
    float * ksfw = (float *) ksb;
    half  * vshw = (half  *) vsb;
    float * vsfw = (float *) vsb;

    float qr[RM][DP];
    float acc[RM][DP];
    float m_run[RM];
    float d_run[RM];
#pragma unroll
    for (int r = 0; r < RM; ++r) {
        const float * qrow = (const float *)(q + (size_t)(qok[r] ? tq[r] : 0)*qnb1 + (size_t) hq*qnb2);
#pragma unroll
        for (int c = 0; c < DP/LOD2_CH; ++c) {
#pragma unroll
            for (int j = 0; j < LOD2_CH; ++j) {
                qr [r][c*LOD2_CH + j] = qok[r] ? qrow[LOD2_DSD(c) + j] : 0.0f;
                acc[r][c*LOD2_CH + j] = 0.0f;
            }
        }
        m_run[r] = -INFINITY;
        d_run[r] = 0.0f;
    }

    const int jend = q0 + min(t0 + TM - 1, nq - 1);
    const int nthr = RG*LOD2_TN;

    for (int j0 = l0; j0 <= jend; j0 += CT) {
        __syncthreads();
        for (int i = tid; i < CT*DPT; i += nthr) {
            const int c = j0 + i/DPT;
            const float x = c <= jend ? lod2_get(k + (size_t) c*knb1 + (size_t) h*knb2, ktype, i%DPT) : 0.0f;
            if (KVH) { kshw[i] = __float2half(x); } else { ksfw[i] = x; }
        }
        for (int i = tid; i < CT*DPT; i += nthr) {
            const int c = j0 + i/DPT;
            const float x = c <= jend ? lod2_get(v + (size_t) c*vnb1 + (size_t) h*vnb2, vtype, i%DPT) : 0.0f;
            if (KVH) { vshw[i] = __float2half(x); } else { vsfw[i] = x; }
        }
        __syncthreads();

        // sc holds the tile's scores and is then overwritten in place by the
        // probabilities: every lane ends up with all CT of them, so the
        // row max and the row sum are register reductions and the value pass
        // needs no shared memory for them.
        float sc[RM][CT];
#pragma unroll
        for (int u = 0; u < CT; ++u) {
            // Two partial sums per query rather than one.  The score pass is
            // the only FMA chain here that is NOT packed - 240 v_fmac_f32
            // against the value pass's 129 v_pk_fma_f32 - and accumulating
            // both halves of a pair into one scalar is the obvious reason.
            // Measured: no change whatsoever.  The disassembly is identical
            // byte for byte, so the compiler already reassociates this and
            // declines to pack it for some other reason.  Kept because it is
            // the clearer statement of intent, not because it bought anything.
            float a0[RM], a1[RM];
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                a0[r] = 0.0f;
                a1[r] = 0.0f;
            }
#pragma unroll
            for (int c = 0; c < DP/LOD2_CH; ++c) {
                const int off = u*DPT + LOD2_DSD(c);
#pragma unroll
                for (int i = 0; i < LOD2_CH/2; ++i) {
                    const float2 x = KVH ? __half22float2(((const half2 *)(ksh + off))[i])
                                         :                ((const float2 *)(ksf + off))[i];
#pragma unroll
                    for (int r = 0; r < RM; ++r) {
                        a0[r] += qr[r][c*LOD2_CH + 2*i    ]*x.x;
                        a1[r] += qr[r][c*LOD2_CH + 2*i + 1]*x.y;
                    }
                }
            }
            float a[RM];
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                a[r] = a0[r] + a1[r];
            }
            const int c = j0 + u;
#pragma unroll
            for (int r = 0; r < RM; ++r) {
#pragma unroll
                for (int o = LOD2_TN/2; o > 0; o >>= 1) {
                    a[r] += __shfl_xor_sync(0xffffffff, a[r], o, LOD2_TN);
                }
                sc[r][u] = (qok[r] && c <= qend[r]) ? a[r]*sl2 : -INFINITY;
            }
        }

        float co[RM];
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            float tmax = -INFINITY;
#pragma unroll
            for (int u = 0; u < CT; ++u) {
                tmax = fmaxf(tmax, sc[r][u]);
            }
            const float nm = fmaxf(m_run[r], tmax);
            co[r] = m_run[r] == -INFINITY ? 0.0f : exp2f(m_run[r] - nm);

            float rs = 0.0f;
#pragma unroll
            for (int u = 0; u < CT; ++u) {
                sc[r][u] = nm == -INFINITY ? 0.0f : exp2f(sc[r][u] - nm);
                rs += sc[r][u];
            }
            d_run[r] = d_run[r]*co[r] + rs;
            m_run[r] = nm;
#pragma unroll
            for (int e = 0; e < DP; ++e) {
                acc[r][e] *= co[r];
            }
        }

#pragma unroll
        for (int u = 0; u < CT; ++u) {
#pragma unroll
            for (int c = 0; c < DP/LOD2_CH; ++c) {
                const int off = u*DPT + LOD2_DSD(c);
#pragma unroll
                for (int e = 0; e < LOD2_CH/2; ++e) {
                    const float2 x = KVH ? __half22float2(((const half2 *)(vsh + off))[e])
                                         :                ((const float2 *)(vsf + off))[e];
#pragma unroll
                    for (int r = 0; r < RM; ++r) {
                        acc[r][c*LOD2_CH + 2*e    ] += sc[r][u]*x.x;
                        acc[r][c*LOD2_CH + 2*e + 1] += sc[r][u]*x.y;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RM; ++r) {
        if (!qok[r]) {
            continue;
        }
        const size_t unit = (size_t) hq*nq + tq[r];
        if (n == 0) {
            lm[unit] = m_run[r];
            ld[unit] = d_run[r];
        }
#pragma unroll
        for (int c = 0; c < DP/LOD2_CH; ++c) {
#pragma unroll
            for (int j = 0; j < LOD2_CH; ++j) {
                lacc[unit*Dv + LOD2_DSD(c) + j] = acc[r][c*LOD2_CH + j];
            }
        }
    }
}

// Coarse branch, tiled.  The per-query form inside lod2_k_attn reads the whole
// slot-mean value table once per query - nq * s_len * Dv floats with no reuse
// whatsoever, which is why rocprof reports group_segment_size 0 for that kernel
// against 34944 for llama.cpp's own flash attention.  A block here owns LOD2_TM
// queries of one query head and walks the slot table in LOD2_TN columns,
// staging each column tile's value means once for all TM queries.  Mean traffic
// therefore drops by LOD2_TM.
//
// The routing scan is replicated verbatim rather than handed over from
// lod2_k_attn: it touches two floats per slot against the coarse branch's Dv,
// so it is cheap next to what the tiling saves, and running the identical scan
// in the identical order keeps the two kernels' route lists bit-identical -
// which is what lets this be verified without touching the merge.
// NACC is Dv/LOD2_TN, a template parameter for the same reason DP is one in
// lod2_k_local_ds: with the runtime Dv the compiler cannot prove that a lane's
// slice is 8-byte aligned and falls back to one ds_read_b32 per element.
template <int TM, int NACC>
static __global__ __launch_bounds__(TM*LOD2_TN) void lod2_k_coarse(
        const float * __restrict__ s_kv, const float * __restrict__ s_mn,
        const float * __restrict__ lgt,
        float * __restrict__ cm, float * __restrict__ cd, float * __restrict__ cacc,
        const int S_cap, const int Ds, const int D, const int Dv, const int g,
        const int nq, const int s_len, const int lstride,
        const int n_routes, const int sink, const float sl2) {
    const int hq  = blockIdx.y;
    const int h   = hq/g;
    const int t0  = blockIdx.x*TM;
    const int tid = threadIdx.x;
    const int m   = tid/LOD2_TN;
    const int n   = tid%LOD2_TN;

    const int tq   = t0 + m;
    const bool qok = tq < nq;

    const float * slot = s_kv + (size_t) h*S_cap*Ds;
    const float * mn   = s_mn + (size_t) h*S_cap*(Ds - 1);
    const float * lg   = lgt  + (size_t)((size_t) hq*nq + (qok ? tq : 0))*lstride;

    const int DvT = NACC*LOD2_TN;              // == Dv, but compile time

    extern __shared__ float csm[];
    float * vs = csm;                          // [TN][Dv]
    float * ps = vs + (size_t) LOD2_TN*DvT;    // [TM][TN]
    float * cs = ps + TM*LOD2_TN;         // [TN]

    // --- routing, byte for byte the scan in lod2_k_attn --------------------
    const int protect = min(sink, s_len);
    const int R       = min(n_routes, max(s_len - protect, 0));

    int   routes[LOD2_MAX_R];
    float rscore[LOD2_MAX_R];
    int   n_open = 0;
#pragma unroll
    for (int r = 0; r < LOD2_MAX_R; ++r) {
        routes[r] = -1;
        rscore[r] = -INFINITY;
    }

    if (qok && R > 0) {
        for (int s = protect; s < s_len; ++s) {
            const float cnt = slot[(size_t) s*Ds + D + Dv];
            if (!(cnt > 0.5f)) {
                continue;
            }
            const float lgv = sl2*lg[s] + log2f(cnt);
            if (n_open < R || lgv > rscore[R - 1]) {
                int pos = min(n_open, R - 1);
                while (pos > 0 && rscore[pos - 1] < lgv) {
                    rscore[pos] = rscore[pos - 1];
                    routes[pos] = routes[pos - 1];
                    pos--;
                }
                rscore[pos] = lgv;
                routes[pos] = s;
                n_open = min(n_open + 1, R);
            }
        }
    }

    // --- coarse -------------------------------------------------------------
    float acc[LOD2_TACC];
#pragma unroll
    for (int e = 0; e < LOD2_TACC; ++e) {
        acc[e] = 0.0f;
    }
    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (int s0 = 0; s0 < s_len; s0 += LOD2_TN) {
        __syncthreads();
        for (int i = tid; i < LOD2_TN; i += TM*LOD2_TN) {
            const int s = s0 + i;
            cs[i] = s < s_len ? slot[(size_t) s*Ds + D + Dv] : 0.0f;
        }
        for (int i = tid; i < LOD2_TN*DvT; i += TM*LOD2_TN) {
            const int nn = i/DvT;
            const int d  = i%DvT;
            const int s  = s0 + nn;
            vs[i] = s < s_len ? mn[(size_t) s*(Ds - 1) + D + d] : 0.0f;
        }
        __syncthreads();

        const int s = s0 + n;
        float lgv = -INFINITY;
        if (qok && s < s_len && cs[n] > 0.5f) {
            bool opened = false;
            for (int r = 0; r < n_open; ++r) {
                opened |= (routes[r] == s);
            }
            if (!opened) {
                lgv = sl2*lg[s] + log2f(cs[n]);
            }
        }

        float tm = lgv;
#pragma unroll
        for (int off = LOD2_TN/2; off > 0; off >>= 1) {
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffff, tm, off, LOD2_TN));
        }

        const float nm = fmaxf(m_run, tm);
        const float co = m_run == -INFINITY ? 0.0f : exp2f(m_run - nm);
        const float pr = nm == -INFINITY ? 0.0f : exp2f(lgv - nm);

        float rs = pr;
#pragma unroll
        for (int off = LOD2_TN/2; off > 0; off >>= 1) {
            rs += __shfl_xor_sync(0xffffffff, rs, off, LOD2_TN);
        }
        d_run = d_run*co + rs;
        m_run = nm;

        ps[m*LOD2_TN + n] = pr;
        __syncthreads();

        // The probabilities come out of shared memory once per tile instead of
        // once per (dimension, slot).  All LOD2_TN lanes of a row want the same
        // sixteen floats, so this is four broadcast reads against 256.
        float pv[LOD2_TN];
#pragma unroll
        for (int u = 0; u < LOD2_TN; ++u) {
            pv[u] = ps[m*LOD2_TN + u];
        }

        // Dimensions are taken in striped pairs - lane n at dword 2n within a
        // run - so a row's sixteen lanes cover thirty-two consecutive dwords,
        // one per bank, at eight bytes each.  The lane-strided form this
        // replaces read one dword per lane per instruction: conflict free but
        // half the width, so twice the instructions.
#pragma unroll
        for (int c = 0; c < NACC/2; ++c) {
            const int dv = c*2*LOD2_TN + n*2;
            float2 a = make_float2(acc[2*c]*co, acc[2*c + 1]*co);
#pragma unroll
            for (int u = 0; u < LOD2_TN; ++u) {
                const float2 x = *(const float2 *)(vs + (size_t) u*DvT + dv);
                a.x += pv[u]*x.x;
                a.y += pv[u]*x.y;
            }
            acc[2*c    ] = a.x;
            acc[2*c + 1] = a.y;
        }
    }

    if (!qok) {
        return;
    }
    const size_t unit = (size_t) hq*nq + tq;
    if (n == 0) {
        cm[unit] = m_run;
        cd[unit] = d_run;
    }
#pragma unroll
    for (int c = 0; c < NACC/2; ++c) {
        const int dv = c*2*LOD2_TN + n*2;
        cacc[unit*Dv + dv    ] = acc[2*c    ];
        cacc[unit*Dv + dv + 1] = acc[2*c + 1];
    }
}

// One warp per (token, KV head, slot): the whole GQA group's logits come out of
// a single read of the slot's mean.
//
// This is the generation counterpart of the routing GEMM.  Both compute the
// same q . mean_k, but a GEMV reads the mean table once per query head, and
// ggml_mul_mat additionally wants a compacted copy of it because the K half of
// a mean row is strided.  Reading per KV head instead divides the traffic by
// the GQA factor and needs no copy at all - the count bias lands here too, so
// lod2_k_bias is not needed on this path either.
template <int W, int EV>
static __global__ void lod2_k_logits_kv(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const float * __restrict__ s_kv, const float * __restrict__ s_mn,
        float * __restrict__ logits,
        const int S_cap, const int Ds, const int D, const int Dv,
        const int nq, const int Hq, const int g, const int s_len, const float sl2) {
    const int w    = blockIdx.x*(blockDim.x/LOD2_LANES) + threadIdx.x/LOD2_LANES;
    const int lane = threadIdx.x % LOD2_LANES;

    const int nh = Hq/g;              // KV heads
    const int s  = w % s_len;
    const int hu = w / s_len;
    if (hu >= nq*nh) {
        return;
    }
    const int t = hu / nh;
    const int h = hu % nh;

    const float cnt = s_kv[((size_t) h*S_cap + s)*Ds + D + Dv];
    if (!(cnt > 0.5f)) {
        if (lane == 0) {
            for (int j = 0; j < g; ++j) {
                logits[(size_t)(t*Hq + h*g + j)*s_len + s] = -INFINITY;
            }
        }
        return;
    }

    const float * mn = s_mn + ((size_t) h*S_cap + s)*(Ds - 1);
    float mk[4*EV];
#pragma unroll
    for (int e = 0; e < EV; ++e) {
        const int i0 = W*(lane + e*LOD2_LANES);
        lod2_ldw<W>(&mk[e*W], mn + i0, i0 + W <= D);
    }

    const float bias = log2f(cnt);
    // Four query heads at a time.  Every head of the group reads the same mean
    // row, and reducing one head's dot before starting the next made the group
    // a chain of g dependent shuffle reductions; taken four at a time the
    // chains interleave.  Each head's sum is still formed in the same order.
    for (int j0 = 0; j0 < g; j0 += 4) {
        float a[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (j0 + u >= g) {
                continue;
            }
            const float * qv = (const float *)(q + (size_t) t*qnb1 + (size_t)(h*g + j0 + u)*qnb2);
#pragma unroll
            for (int e = 0; e < EV; ++e) {
#pragma unroll
                for (int c = 0; c < W; ++c) {
                    const int i = W*(lane + e*LOD2_LANES) + c;
                    a[u] += i < D ? qv[i]*mk[e*W + c] : 0.0f;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            a[u] = lod2_warp_sum(a[u]);
        }
        if (lane == 0) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (j0 + u < g) {
                    logits[(size_t)(t*Hq + h*g + j0 + u)*s_len + s] = sl2*a[u] + bias;
                }
            }
        }
    }
}

// one warp per (unit, slot): logit = scale*dot(q, mean_k) + log2(count)
static __global__ void lod2_k_logits(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const float * __restrict__ s_kv, const float * __restrict__ s_mn,
        float * __restrict__ logits,
        const int S_cap, const int Ds, const int D, const int Dv,
        const int nq, const int Hq, const int g, const int s_len, const float sl2) {
    const int w = blockIdx.x*(blockDim.x/LOD2_LANES) + threadIdx.x/LOD2_LANES;
    const int lane = threadIdx.x % LOD2_LANES;

    const int s    = w % s_len;
    const int unit = w / s_len;
    if (unit >= nq*Hq) {
        return;
    }
    const int t  = unit / Hq;
    const int hq = unit % Hq;
    const int h  = hq / g;

    const float * qv  = (const float *)(q + (size_t) t*qnb1 + (size_t) hq*qnb2);
    const float * row = s_kv + ((size_t) h*S_cap + s)*Ds;
    const float * mn  = s_mn + ((size_t) h*S_cap + s)*(Ds - 1);

    const float cnt = row[D + Dv];
    float lg = -INFINITY;
    if (cnt > 0.5f) {
        float a = 0.0f;
        for (int i = lane; i < D; i += LOD2_LANES) {
            a += qv[i]*mn[i];
        }
        a = lod2_warp_sum(a);
        lg = sl2*a + log2f(cnt);
    }
    if (lane == 0) {
        logits[(size_t) unit*s_len + s] = lg;
    }
}

// One block per unit.  Every thread keeps a private top-R over its stride, then
// the candidate lists are merged.
//
// The merge is what this kernel costs.  A block-wide tree needs a __syncthreads
// per level, so R rounds came to ~64 barriers, and with one block per unit only
// 32 of 304 CUs are occupied - there is nothing to overlap those barriers with,
// so they are paid at full latency (359 us per dispatch for the thread-0 walk
// it replaced, 46 us for the tree).  Merging inside a warp instead costs
// shuffles and no barriers: each warp reduces its 64 lists to R, one barrier
// publishes the LOD2_WPB survivors, and warp 0 merges those.  Slots are unique
// across threads (disjoint strides), so a lane knows it won a round by matching
// the winning slot index and needs no separate owner channel.
static __global__ __launch_bounds__(LOD2_BLOCK) void lod2_k_topk(
        float * __restrict__ logits, int * __restrict__ top,
        float * __restrict__ cmax,
        const int nq, const int Hq, const int s_len, const int protect, const int R) {
    const int unit = blockIdx.x;
    if (unit >= nq*Hq) {
        return;
    }
    float * lg  = logits + (size_t) unit*s_len;
    int   * out = top    + (size_t) unit*LOD2_MAX_R;

    const int wp   = threadIdx.x/LOD2_LANES;
    const int lane = threadIdx.x % LOD2_LANES;

    // The sink slots are never opened, so they stay in the coarse branch and
    // their maximum belongs in cmax.
    float pmax = -INFINITY;
    for (int s = threadIdx.x; s < protect && s < s_len; s += blockDim.x) {
        pmax = fmaxf(pmax, lg[s]);
    }

    // Keeping R+1 rather than R is what folds the old lod2_k_prep away: the
    // largest logit the coarse branch keeps is exactly the (R+1)-th largest
    // here, so the branch maximum comes out of this merge instead of a second
    // full pass over the logits.
    float best[LOD2_TOPK_N];
    int   barg[LOD2_TOPK_N];
#pragma unroll
    for (int r = 0; r < LOD2_TOPK_N; ++r) {
        best[r] = -INFINITY;
        barg[r] = -1;
    }

    const int N = min(R + 1, LOD2_TOPK_N);
    // Four logits in flight.  Sinking a candidate through the sorted list is a
    // dependent chain, so a one-at-a-time loop made every load wait for the
    // previous insertion: at 32k that is eleven memory round trips per thread
    // and it was the whole cost of this phase.  The four are still inserted in
    // ascending slot order, so the list - ties included - is unchanged.
    for (int s0 = protect + threadIdx.x; s0 < s_len; s0 += 4*blockDim.x) {
        float vv[4];
        int   ii[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int s = s0 + u*blockDim.x;
            ii[u] = s;
            vv[u] = s < s_len ? lg[s] : -INFINITY;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (ii[u] >= s_len) {
                continue;
            }
            // Sink the new candidate through the sorted list with constant
            // indices.  The old form skipped out early and then shifted with a
            // runtime index; that made the body control-dependent on its own
            // load as well.
            float v = vv[u];
            int   i = ii[u];
#pragma unroll
            for (int r = 0; r < LOD2_TOPK_N; ++r) {
                if (r < N) {
                    const bool ins = v > best[r] || (v == best[r] && i < barg[r]);
                    const float ov = best[r];
                    const int   oi = barg[r];
                    best[r] = ins ? v  : ov;
                    barg[r] = ins ? i  : oi;
                    v       = ins ? ov : v;
                    i       = ins ? oi : i;
                }
            }
        }
    }

    __shared__ float sh_val[LOD2_WPB*LOD2_TOPK_N];
    __shared__ int   sh_idx[LOD2_WPB*LOD2_TOPK_N];
    __shared__ float sh_pmx[LOD2_WPB];

#pragma unroll
    for (int m = LOD2_LANES/2; m > 0; m >>= 1) {
        pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffff, pmax, m, LOD2_LANES));
    }
    if (lane == 0) {
        sh_pmx[wp] = pmax;
    }

    // warp round: the lists are sorted, so only their heads can win
    int cur = 0;
    for (int r = 0; r < LOD2_TOPK_N; ++r) {
        const float v = cur < N ? best[cur] : -INFINITY;
        const int   s = cur < N ? barg[cur] : -1;
        float bv = v;
        int   bs = s;
        lod2_warp_argmax(bv, bs);
        if (lane == 0) {
            sh_val[wp*LOD2_TOPK_N + r] = bv;
            sh_idx[wp*LOD2_TOPK_N + r] = bs;
        }
        if (bs >= 0 && s == bs) {
            cur++;
        }
    }
    __syncthreads();

    // final round: LOD2_WPB*(R+1) candidates, one per lane, each won at most once
    if (wp == 0) {
        const int n = LOD2_WPB*LOD2_TOPK_N;
        float v = lane < n ? sh_val[lane] : -INFINITY;
        int   s = lane < n ? sh_idx[lane] : -1;

        float cm = -INFINITY;
        for (int u = 0; u < LOD2_WPB; ++u) {
            cm = fmaxf(cm, sh_pmx[u]);
        }

        for (int r = 0; r < LOD2_TOPK_N; ++r) {
            float bv = v;
            int   bs = s;
            lod2_warp_argmax(bv, bs);
            if (r < R) {
                // opened: the routed branch re-reads it exactly, so -inf here
                // makes the coarse branch skip it without a per-slot test
                if (lane == 0) {
                    out[r] = bs;
                    if (bs >= 0) {
                        lg[bs] = -INFINITY;
                    }
                }
            } else if (r == R) {
                cm = fmaxf(cm, bv);
            }
            if (bs >= 0 && s == bs) {
                v = -INFINITY;
                s = -1;
            }
        }
        if (lane == 0) {
            for (int r = R; r < LOD2_MAX_R; ++r) {
                out[r] = -1;
            }
            cmax[unit] = cm;
        }
    }

}

// One warp per (unit, route): which page of the routed slot the exact columns
// come from.  It is one argmax over a slot's pages and every split of that
// route needs the same answer, so leaving it in lod2_k_part confined the whole
// routed branch to the R splits that owned a route.  It does not belong in
// lod2_k_topk either: measured on 27B/32k generation the scan costs 0.7 us of
// work there but 20.7 us of kernel time, because that block has one block per
// unit and 24 of 304 CUs busy.  On its own grid it is 192 blocks.
static __global__ void lod2_k_route(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const float * __restrict__ p_kv, const int * __restrict__ s_pg,
        const int * __restrict__ top, int * __restrict__ sel,
        const int P_cap, const int S_cap, const int Ds, const int D, const int Dv,
        const int ps, const int pps, const int g_kv,
        const int nq, const int Hq, const float sl2) {
    const int r    = blockIdx.x;
    const int unit = blockIdx.y;
    const int lane = threadIdx.x % LOD2_LANES;
    const int wp   = threadIdx.x / LOD2_LANES;
    const int nw   = blockDim.x  / LOD2_LANES;
    if (unit >= nq*Hq) {
        return;
    }

    const int s = top[(size_t) unit*LOD2_MAX_R + r];

    int   bestp = -1;
    int   besto = INT_MAX;
    float best  = -INFINITY;
    if (s >= 0) {
        const int t  = unit / Hq;
        const int hq = unit % Hq;
        const int h  = hq / g_kv;

        const float * qv   = (const float *)(q + (size_t) t*qnb1 + (size_t) hq*qnb2);
        const float * page = p_kv + (size_t) h*P_cap*Ds;
        const int   * ent  = s_pg + (size_t) h*S_cap*(pps + 1) + (size_t) s*(pps + 1);

        const int n_pg = (ent[pps] + ps - 1)/ps;
        // The warps of the block take the pages in turn and four are in flight
        // within a warp, so a slot with many pages is not a chain of whole-row
        // reads owned by one wave.
        for (int o0 = 4*wp; o0 < n_pg; o0 += 4*nw) {
            int   pg [4];
            float cnt[4];
            float aa [4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                pg [u] = o0 + u < n_pg ? ent[o0 + u] : -1;
                cnt[u] = 0.0f;
                aa [u] = 0.0f;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (pg[u] >= 0) {
                    cnt[u] = page[(size_t) pg[u]*Ds + D + Dv];
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (pg[u] >= 0 && cnt[u] > 0.0f) {
                    const float * prow = page + (size_t) pg[u]*Ds;
                    float x = 0.0f;
                    for (int i = lane; i < D; i += LOD2_LANES) {
                        x += qv[i]*prow[i];
                    }
                    aa[u] = x;
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                aa[u] = lod2_warp_sum(aa[u]);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (pg[u] < 0 || !(cnt[u] > 0.0f)) {
                    continue;
                }
                const float sc = sl2*(aa[u]/cnt[u]) + log2f(cnt[u]);
                if (sc > best || (sc == best && o0 + u < besto)) {
                    best  = sc;
                    besto = o0 + u;
                    bestp = pg[u];
                }
            }
        }
    }

    // The scan is ascending in o and takes a page only on a strictly greater
    // score, so the winner is the lowest o among equals however the pages were
    // handed out.  The block merge keeps that rule.
    __shared__ float sh_v[LOD2_BLOCK/LOD2_LANES];
    __shared__ int   sh_o[LOD2_BLOCK/LOD2_LANES];
    __shared__ int   sh_p[LOD2_BLOCK/LOD2_LANES];
    if (lane == 0) {
        sh_v[wp] = best;
        sh_o[wp] = besto;
        sh_p[wp] = bestp;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int u = 1; u < nw; ++u) {
            if (sh_v[u] > best || (sh_v[u] == best && sh_o[u] < besto)) {
                best  = sh_v[u];
                besto = sh_o[u];
                bestp = sh_p[u];
            }
        }
        sel[(size_t) unit*LOD2_MAX_R + r] = bestp;
    }
}

// One warp per (token, KV head, split), covering all G query heads of the GQA
// group at once.  The four query heads of a group read the *same* slot rows and
// the same local K/V columns; giving each its own warp fetched every row four
// times, and the slot scan was the largest generation kernel at 2.45 ms/token.
// Here the row is read once and applied to G accumulators.
//
// The partials keep the per-query-head layout, so lod2_k_reduce is unchanged.
//
// W is the load width in floats per lane and EV the number of such loads that
// cover one value head; together they fix the lane -> element map used by every
// branch's accumulator.
template <int G, int W, int EV>
static __global__ void lod2_k_part(
        const char * __restrict__ q, const size_t qnb1, const size_t qnb2,
        const char * __restrict__ k, const int ktype, const size_t knb1, const size_t knb2,
        const char * __restrict__ v, const int vtype, const size_t vnb1, const size_t vnb2,
        const float * __restrict__ s_kv, const float * __restrict__ s_mn,
        const float * __restrict__ p_kv,
        const int * __restrict__ p_idx, const int * __restrict__ s_pg,
        const float * __restrict__ logits, const int * __restrict__ top,
        const float * __restrict__ cmax, const int * __restrict__ rsel,
        float * __restrict__ pm, float * __restrict__ pd, float * __restrict__ pacc,
        const int S_cap, const int P_cap, const int Ds, const int D, const int Dv,
        const int ps, const int pps, const int g_kv,
        const int l0, const int * __restrict__ q0p, const int nq, const int Hq,
        const int s_len, const int NS, const float sl2, const int dbr) {
    const int q0   = q0p[0];
    const int wp   = threadIdx.x/LOD2_LANES;   // warp within the block
    const int lane = threadIdx.x % LOD2_LANES;

    // A block owns LOD2_WPB consecutive splits of one unit, so its warps can
    // fold their partials together in LDS before anything reaches memory.  The
    // split count and the reduce cost used to be the same number: more splits
    // meant more parallelism here but a proportionally longer serial fold in
    // lod2_k_reduce, which pinned NS to a value that left this kernel with four
    // waves per CU and no way to hide the load latency.  Folding per block
    // divides what the reduce sees by LOD2_WPB and lets NS grow on its own.
    const int groups = Hq/G;
    const int nw  = blockDim.x/LOD2_LANES;   // splits folded by this block
    const int sp  = blockIdx.x*nw + wp;
    const int kvu = blockIdx.y;
    if (kvu >= nq*groups) {
        return;
    }
    const bool live   = sp < NS;
    const int t       = kvu / groups;
    const int hq_base = (kvu % groups)*G;
    const int h       = hq_base / g_kv; // KV head the whole warp reads

    const float * qv[G];
    const float * lg[G];
    const int   * rt[G];
    float         m0[G];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int hq   = hq_base + j;
        const int unit = t*Hq + hq;
        qv[j] = (const float *)(q + (size_t) t*qnb1 + (size_t) hq*qnb2);
        lg[j] = logits + (size_t) unit*s_len;
        rt[j] = top + (size_t) unit*LOD2_MAX_R;
        m0[j] = cmax[unit];
    }

    const float * slot = s_kv + (size_t) h*S_cap*Ds;
    const float * mean = s_mn + (size_t) h*S_cap*(Ds - 1);

    // lane -> element map, shared by every branch's accumulator
    bool ok[EV];
#pragma unroll
    for (int e = 0; e < EV; ++e) {
        ok[e] = W*(lane + e*LOD2_LANES) + W <= Dv;
    }

    float acc[G][3][W*EV];
    float mm[G][3], dd[G][3];
#pragma unroll
    for (int j = 0; j < G; ++j) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
            mm[j][b] = -INFINITY;
            dd[j][b] = 0.0f;
#pragma unroll
            for (int c = 0; c < W*EV; ++c) {
                acc[j][b][c] = 0.0f;
            }
        }
        // the coarse maximum is fixed for the whole branch, splits included
        mm[j][0] = live ? m0[j] : -INFINITY;
    }

    // branch 0: coarse.  lod2_k_topk finished the logits - count bias applied,
    // opened slots set to -inf - and handed over the branch maximum, so this is
    // a plain weighted sum over the value half of the mean table.  There is no
    // running maximum to carry, which is the whole point: the old form fed each
    // slot's rescale into the next one's accumulator and the loop ran at the
    // latency of a dependent exp2f chain (~2% of achievable bandwidth) however
    // well the reads coalesced.  Nothing here depends on the previous slot, so
    // the loads pipeline.
    if (live && (dbr & 1)) {
        const int lo = (int)(((long) s_len*sp)/NS);
        const int hi = (int)(((long) s_len*(sp + 1))/NS);

        // an all -inf branch would give exp2f(-inf - -inf) = NaN; with the
        // shift at 0 every logit is -inf, so p is 0 and the branch drops out
        // through dd == 0 on its own
        float mj[G];
#pragma unroll
        for (int j = 0; j < G; ++j) {
            mj[j] = m0[j] > -INFINITY ? m0[j] : 0.0f;
        }

        for (int s = lo; s < hi; ++s) {
            const float * vrow = mean + (size_t) s*(Ds - 1) + D;
            float val[4*EV];
#pragma unroll
            for (int e = 0; e < EV; ++e) {
                lod2_ldw<W>(&val[e*W], vrow + W*(lane + e*LOD2_LANES), ok[e]);
            }
#pragma unroll
            for (int j = 0; j < G; ++j) {
                const float p = exp2f(lg[j][s] - mj[j]);
                dd[j][0] += p;
#pragma unroll
                for (int c = 0; c < W*EV; ++c) {
                    acc[j][0][c] += p*val[c];
                }
            }
        }
    }

    // branch 1: local window.  One K column and one V column per position,
    // shared by the group.
    if (live && (dbr & 2)) {
        const int len = q0 + t - l0 + 1;
        const int lo  = l0 + (int)(((long) len*sp)/NS);
        const int hi  = l0 + (int)(((long) len*(sp + 1))/NS);
        for (int c = lo; c < hi; ++c) {
            const char * krow = k + (size_t) c*knb1 + (size_t) h*knb2;
            const char * vrow = v + (size_t) c*vnb1 + (size_t) h*vnb2;

            // K is read once for the whole group and folded straight into the
            // partial dot; the query head count never touches memory traffic.
            float qk[G];
#pragma unroll
            for (int j = 0; j < G; ++j) {
                qk[j] = 0.0f;
            }
            for (int i = lane; i < D; i += LOD2_LANES) {
                const float kx = lod2_get(krow, ktype, i);
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    qk[j] += qv[j][i]*kx;
                }
            }

            float vv[4*EV];
#pragma unroll
            for (int e = 0; e < EV; ++e) {
#pragma unroll
                for (int c2 = 0; c2 < W; ++c2) {
                    const int i = W*(lane + e*LOD2_LANES) + c2;
                    vv[e*W + c2] = i < Dv ? lod2_get(vrow, vtype, i) : 0.0f;
                }
            }
#pragma unroll
            for (int j = 0; j < G; ++j) {
                const float a = lod2_warp_sum(qk[j])*sl2;
                const float nm = fmaxf(mm[j][1], a);
                const float co = mm[j][1] == -INFINITY ? 0.0f : exp2f(mm[j][1] - nm);
                const float pr = exp2f(a - nm);
                dd[j][1] = dd[j][1]*co + pr;
#pragma unroll
                for (int c2 = 0; c2 < W*EV; ++c2) {
                    acc[j][1][c2] = acc[j][1][c2]*co + pr*vv[c2];
                }
                mm[j][1] = nm;
            }
        }
    }

    // branch 2: routed leaves.  The routes differ per query head, so there is
    // no row to share between them.  It used to run entirely in split 0: a page
    // scan plus a residual plus 16 exact columns per route, all warp reductions
    // back to back, which made that one warp several times longer than the
    // splits that only walked their slice of slots - and the kernel ends when
    // the last warp does.  Handing route r to split r spread it over R splits,
    // which was still 8 of ~200: measured on 27B/32k generation the branch was
    // 74% of this kernel, and of that the exact columns were 88%.
    //
    // A route's page is now chosen in lod2_k_topk, so a split no longer needs
    // the scan to know which columns it would read.  That makes the residual
    // and every exact column a work item of its own, and the branch spreads
    // over R*(ps + 1) splits instead of R.  The branch is merged by logsumexp,
    // which is associative, so which split an item lands in does not change
    // what the merge computes.
    if (live && (dbr & 4)) {
        const float * page = p_kv + (size_t) h*P_cap*Ds;
        const int   * pidx = p_idx + (size_t) h*P_cap*ps;

        const int NW = ps + 1;   // the residual, then the page's exact columns

#pragma unroll
        for (int j = 0; j < G; ++j) {
            const int unit = t*Hq + hq_base + j;
            for (int it = sp; it < LOD2_MAX_R*NW; it += NS) {
                const int r   = it/NW;
                const int sub = it - r*NW;
                if (rt[j][r] < 0) {
                    continue;
                }
                const int selp = rsel[(size_t) unit*LOD2_MAX_R + r];
                if (selp < 0) {
                    continue;
                }
                const float * prow  = page + (size_t) selp*Ds;
                const float   p_cnt = prow[D + Dv];

                if (sub == 0) {
                    const float * srow  = slot + (size_t) rt[j][r]*Ds;
                    const float   r_cnt = srow[D + Dv] - p_cnt;
                    if (!(r_cnt > 0.0f)) {
                        continue;
                    }
                    const float inv = 1.0f/r_cnt;
                    float a = 0.0f;
                    for (int i = lane; i < D; i += LOD2_LANES) {
                        a += qv[j][i]*(srow[i] - prow[i]);
                    }
                    a = lod2_warp_sum(a);
                    const float x  = sl2*(a*inv) + log2f(r_cnt);
                    const float nm = fmaxf(mm[j][2], x);
                    const float co = mm[j][2] == -INFINITY ? 0.0f : exp2f(mm[j][2] - nm);
                    const float pr = exp2f(x - nm);
                    dd[j][2] = dd[j][2]*co + pr;
#pragma unroll
                    for (int e = 0; e < EV; ++e) {
#pragma unroll
                        for (int c2 = 0; c2 < W; ++c2) {
                            const int i = W*(lane + e*LOD2_LANES) + c2;
                            acc[j][2][e*W + c2] = acc[j][2][e*W + c2]*co +
                                (i < Dv ? pr*(srow[D + i] - prow[D + i])*inv : 0.0f);
                        }
                    }
                    mm[j][2] = nm;
                    continue;
                }

                const int i = sub - 1;
                if (i >= min((int) p_cnt, ps)) {
                    continue;
                }
                const int col = pidx[(size_t) selp*ps + i];
                if (col < 0) {
                    continue;
                }
                const char * krow = k + (size_t) col*knb1 + (size_t) h*knb2;
                const char * vrow = v + (size_t) col*vnb1 + (size_t) h*vnb2;
                float a = 0.0f;
                for (int d = lane; d < D; d += LOD2_LANES) {
                    a += qv[j][d]*lod2_get(krow, ktype, d);
                }
                a = lod2_warp_sum(a)*sl2;
                const float nm = fmaxf(mm[j][2], a);
                const float co = mm[j][2] == -INFINITY ? 0.0f : exp2f(mm[j][2] - nm);
                const float pr = exp2f(a - nm);
                dd[j][2] = dd[j][2]*co + pr;
#pragma unroll
                for (int e = 0; e < EV; ++e) {
#pragma unroll
                    for (int c2 = 0; c2 < W; ++c2) {
                        const int d = W*(lane + e*LOD2_LANES) + c2;
                        acc[j][2][e*W + c2] = acc[j][2][e*W + c2]*co +
                            (d < Dv ? pr*lod2_get(vrow, vtype, d) : 0.0f);
                    }
                }
                mm[j][2] = nm;
            }
        }
    }

    // Fold this block's LOD2_WPB splits into one partial.  Merging partial
    // softmaxes needs no running chain: against the common maximum both the
    // denominator and the accumulator are plain weighted sums.
    // One branch at a time: staging all three at once would need
    // LOD2_WPB*G*3*Dv floats, which overflows LDS as soon as a warp covers a
    // whole GQA group, and costs occupancy even when it fits.
    extern __shared__ float smem[];
    float * sacc = smem;                          // [nw][G][Dv], reused per branch
    float * sm   = sacc + (size_t) nw*G*Dv;       // [nw][G][3]
    float * sd   = sm   + nw*G*3;

#pragma unroll
    for (int j = 0; j < G; ++j) {
        if (lane == 0) {
#pragma unroll
            for (int b2 = 0; b2 < 3; ++b2) {
                sm[(wp*G + j)*3 + b2] = mm[j][b2];
                sd[(wp*G + j)*3 + b2] = dd[j][b2];
            }
        }
    }

    const int i = threadIdx.x;   // one value dimension per thread
#pragma unroll
    for (int b = 0; b < 3; ++b) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < G; ++j) {
            float * o = sacc + (size_t)(wp*G + j)*Dv;
#pragma unroll
            for (int e = 0; e < EV; ++e) {
#pragma unroll
                for (int c = 0; c < W; ++c) {
                    const int ix = W*(lane + e*LOD2_LANES) + c;
                    if (ix < Dv) {
                        o[ix] = acc[j][b][e*W + c];
                    }
                }
            }
        }
        __syncthreads();

        for (int j = 0; j < G; ++j) {
            float m = -INFINITY;
            for (int u = 0; u < nw; ++u) {
                const int slotp = (u*G + j)*3 + b;
                if (sd[slotp] > 0.0f) {
                    m = fmaxf(m, sm[slotp]);
                }
            }
            float d = 0.0f, a = 0.0f;
            if (m > -INFINITY) {
                for (int u = 0; u < nw; ++u) {
                    const int slotp = (u*G + j)*3 + b;
                    const float wgt = exp2f(sm[slotp] - m);
                    d += wgt*sd[slotp];
                    if (i < Dv) {
                        a += wgt*sacc[(size_t)(u*G + j)*Dv + i];
                    }
                }
            }
            const size_t base = (size_t)((t*Hq + hq_base + j)*gridDim.x + blockIdx.x)*3 + b;
            if (i == 0) {
                pm[base] = m;
                pd[base] = d;
            }
            if (i < Dv) {
                pacc[base*Dv + i] = a;
            }
        }
    }
}

static __global__ __launch_bounds__(LOD2_RED_BLOCK) void lod2_k_reduce(
        const float * __restrict__ pm, const float * __restrict__ pd,
        const float * __restrict__ pacc,
        char * __restrict__ dst, const size_t dnb1, const size_t dnb2,
        const int nq, const int Hq, const int Dv, const int NS) {
    const int unit = blockIdx.x;
    if (unit >= nq*Hq) {
        return;
    }
    const int t  = unit / Hq;
    const int hq = unit % Hq;

    // Merging partial softmaxes needs no running chain: with
    // m = max_sp m_sp, both the denominator and the accumulator are plain
    // weighted sums with weight exp2(m_sp - m).  Two passes, no dependency.
    //
    // One block per unit and only 32 units means 32 of 304 CUs, so the three
    // branches are walked concurrently rather than one after another: thread
    // (b, i) owns one branch and one value dimension, which cuts the serial
    // length of the fold to a third for free.
    const int b = threadIdx.x/Dv;
    const int i = threadIdx.x % Dv;

    extern __shared__ float rmem[];
    float * sout = rmem;        // [3][Dv]
    float * slse = rmem + 3*Dv; // [3]

    if (b < 3) {
        float m = -INFINITY;
#pragma unroll 8
        for (int sp = 0; sp < NS; ++sp) {
            const size_t idx = (size_t)(unit*NS + sp)*3 + b;
            if (pd[idx] > 0.0f) {
                m = fmaxf(m, pm[idx]);
            }
        }

        // No early-out on an empty partial: skipping it made every iteration
        // control-dependent on its own load, so the fold ran one round trip at
        // a time.  An empty partial has pm == -inf, and exp2f(-inf - m) is 0
        // against any finite m, so weighting it in costs nothing and lets the
        // loads pipeline.  pacc is always written, so there is no garbage to
        // multiply by zero.
        float d = 0.0f, a = 0.0f;
        if (m > -INFINITY) {
            // Eight partials in flight rather than four: with one block per
            // unit there is nothing else resident to hide the fold behind, so
            // its length in memory round trips is its cost.
#pragma unroll 8
            for (int sp = 0; sp < NS; ++sp) {
                const size_t idx = (size_t)(unit*NS + sp)*3 + b;
                const float w = exp2f(pm[idx] - m);
                d += w*pd[idx];
                a += w*pacc[idx*Dv + i];
            }
        }
        sout[b*Dv + i] = d > 0.0f ? a/d : 0.0f;
        if (i == 0) {
            slse[b] = d > 0.0f ? m + log2f(d) : -INFINITY;
        }
    }
    __syncthreads();

    if (threadIdx.x < Dv) {
        const float mx = fmaxf(slse[0], fmaxf(slse[1], slse[2]));
        float wsum = 0.0f;
        float wt[3];
        for (int c = 0; c < 3; ++c) {
            wt[c] = slse[c] > -INFINITY ? exp2f(slse[c] - mx) : 0.0f;
            wsum += wt[c];
        }
        float * o = (float *)(dst + (size_t) hq*dnb1 + (size_t) t*dnb2);
        o[threadIdx.x] = (wt[0]*sout[threadIdx.x] +
                          wt[1]*sout[Dv + threadIdx.x] +
                          wt[2]*sout[2*Dv + threadIdx.x])/wsum;
    }
}

// ---------------------------------------------------------------------------
// launchers

static int lod2_type_code(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return LOD2_T_F32;
        case GGML_TYPE_F16:  return LOD2_T_F16;
        case GGML_TYPE_Q8_0: return LOD2_T_Q8_0;
        default:             return -1;
    }
}

bool ggml_cuda_lod2_supported(const ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[dst->op == GGML_OP_LOD2_ATTN ? 1 : 0];
    const ggml_tensor * v = dst->src[dst->op == GGML_OP_LOD2_ATTN ? 2 : 1];

    if (lod2_type_code(k->type) < 0 || lod2_type_code(v->type) < 0) {
        return false;
    }
    if (dst->op == GGML_OP_LOD2_ATTN) {
        const ggml_tensor * q = dst->src[0];
        // the accumulators are per-lane arrays sized at compile time
        if (v->ne[0] > 16*LOD2_LANES || q->ne[0] > 16*LOD2_LANES) {
            return false;
        }
        if (q->ne[0] + v->ne[0] > LOD2_MAX_ROW) {
            return false; // the prefill staging buffer holds one row
        }
        if (ggml_get_op_params_i32(dst, 2) > LOD2_MAX_R) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_lod2_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k     = dst->src[0];
    const ggml_tensor * v     = dst->src[1];
    ggml_tensor       * s_kv  = dst->src[2];
    ggml_tensor       * p_kv  = dst->src[3];
    ggml_tensor       * p_idx = dst->src[4];
    ggml_tensor       * s_pg  = dst->src[5];
    ggml_tensor       * meta  = dst->src[6];
    ggml_tensor       * s_mn  = dst->src[7];

    const int p0        = ggml_get_op_params_i32(dst, 0);
    const int p1        = ggml_get_op_params_i32(dst, 1);
    const int s_len     = ggml_get_op_params_i32(dst, 2);
    const int s_len_new = ggml_get_op_params_i32(dst, 3);
    const int sink      = ggml_get_op_params_i32(dst, 4);

    const int D     = k->ne[0];
    const int Dv    = v->ne[0];
    const int Ds    = s_kv->ne[0];
    const int Hkv   = k->ne[2];
    const int S_cap = s_kv->ne[1];
    const int P_cap = p_kv->ne[1];
    const int ps    = p_idx->ne[0];
    const int pps   = s_pg->ne[0] - 1;

    const int M        = p1 - p0;
    const int n_append = s_len_new - s_len;

    // padding block: the graph carries a fixed number of these so its node
    // count is stable, and an empty one has nothing to absorb
    if (M == 0) {
        return;
    }

    const int ktype = lod2_type_code(k->type);
    const int vtype = lod2_type_code(v->type);

    cudaStream_t stream = ctx.stream();

    const char * kd = (const char *) k->data;
    const char * vd = (const char *) v->data;
    float * sd = (float *) s_kv->data;
    float * pd = (float *) p_kv->data;
    int   * id = (int   *) p_idx->data;
    int   * gd = (int   *) s_pg->data;
    int   * md = (int   *) meta->data;

    ggml_cuda_pool_alloc<float> best_score(ctx.pool(), (size_t) Hkv*M);
    ggml_cuda_pool_alloc<float> sel       (ctx.pool(), (size_t) Hkv*M);
    ggml_cuda_pool_alloc<int>   best_slot (ctx.pool(), (size_t) Hkv*M);
    ggml_cuda_pool_alloc<int>   is_app    (ctx.pool(), (size_t) Hkv*M);
    ggml_cuda_pool_alloc<int>   app_list  (ctx.pool(), (size_t) Hkv*M);
    ggml_cuda_pool_alloc<int>   owner     (ctx.pool(), (size_t) Hkv*M);

    if (s_len == 0) {
        // seed: every column of the first block becomes its own slot
        GGML_ASSERT(n_append == M);
        lod2_k_append<<<dim3(M, Hkv), LOD2_BLOCK, 0, stream>>>(
                kd, ktype, k->nb[1], k->nb[2], vd, vtype, v->nb[1], v->nb[2],
                sd, S_cap, Ds, D, Dv, p0, M, s_len, nullptr, true);

        lod2_k_iota<<<Hkv, LOD2_BLOCK, 0, stream>>>(owner.get(), M);
    } else {
        GGML_ASSERT(s_len > sink);

        // Columns per block, and the widest that fits shared memory.  The
        // scan over the slot table is shared by all of them, so this divides
        // the kernel's traffic by MT outright.
        // Swept on 27B/32k, total sim time: (MT, block) 1/256 614.8, 1/512
        // 509.8, 4/256 550.6, 4/512 438.8, 8/512 483.0, 16/512 678.6 ms.
        // Sharing the scan helps far less than its 4x traffic cut suggests -
        // the kernel is paced by the per-(column, slot) warp reduction, not by
        // the table - and past four columns the register file gives it back.
        const int simb = getenv("LLAMA_LOD2_SIMB") ? std::max(64, atoi(getenv("LLAMA_LOD2_SIMB"))) : 512;
        const int simw = simb/LOD2_LANES;
        const int simx = getenv("LLAMA_LOD2_SIMMT") ? std::max(1, atoi(getenv("LLAMA_LOD2_SIMMT"))) : 4;
        int mt = 1;
        for (int t = 1; t <= simx && t <= 16; t *= 2) {
            if ((size_t)(t*D + 3*t*simw)*sizeof(float) <= 64*1024) {
                mt = t;
            }
        }
        const size_t sim_lds = (size_t)(mt*D + 3*mt*simw)*sizeof(float);
#define LOD2_LAUNCH_SIM(MT) \
        lod2_k_sim<MT><<<dim3((M + (MT) - 1)/(MT), Hkv), simb, sim_lds, stream>>>( \
                kd, ktype, k->nb[1], k->nb[2], sd, S_cap, Ds, D, Dv, \
                p0, M, s_len, sink, best_score.get(), best_slot.get(), sel.get())
        switch (mt) {
            case 16: LOD2_LAUNCH_SIM(16); break;
            case  8: LOD2_LAUNCH_SIM(8);  break;
            case  4: LOD2_LAUNCH_SIM(4);  break;
            case  2: LOD2_LAUNCH_SIM(2);  break;
            default: LOD2_LAUNCH_SIM(1);  break;
        }
#undef LOD2_LAUNCH_SIM

        int pow2 = 1;
        while (pow2 < M) {
            pow2 <<= 1;
        }
        const int nth = std::min(1024, std::max(LOD2_LANES, pow2/2));
        GGML_ASSERT(pow2 <= 2*nth && "LoD2 update block is too wide to sort in one pass");
        lod2_k_split<<<Hkv, nth, (size_t) 2*nth*(sizeof(float) + sizeof(int)), stream>>>(
                sel.get(), M, n_append, is_app.get(), app_list.get(), owner.get(), s_len);

        if (n_append > 0) {
            lod2_k_append<<<dim3(n_append, Hkv), LOD2_BLOCK, 0, stream>>>(
                    kd, ktype, k->nb[1], k->nb[2], vd, vtype, v->nb[1], v->nb[2],
                    sd, S_cap, Ds, D, Dv, p0, M, s_len, app_list.get(), false);
        }

        lod2_k_dest<<<dim3(M, Hkv), LOD2_BLOCK, D*sizeof(float), stream>>>(
                kd, ktype, k->nb[1], k->nb[2], D, p0, M, s_len, n_append,
                best_score.get(), best_slot.get(), is_app.get(), app_list.get(), owner.get());

        lod2_k_merge<<<dim3(s_len_new, Hkv), LOD2_BLOCK, 0, stream>>>(
                kd, ktype, k->nb[1], k->nb[2], vd, vtype, v->nb[1], v->nb[2],
                sd, S_cap, Ds, D, Dv, p0, M, is_app.get(), owner.get());
    }

    lod2_k_pages<<<dim3(s_len_new, Hkv), LOD2_BLOCK, 0, stream>>>(
            kd, ktype, k->nb[1], k->nb[2], vd, vtype, v->nb[1], v->nb[2],
            pd, id, gd, md, S_cap, P_cap, Ds, D, Dv, ps, pps, p0, M, owner.get());

    lod2_k_means<<<dim3(s_len_new, Hkv), LOD2_BLOCK, 0, stream>>>(
            sd, (float *) s_mn->data, S_cap, Ds, D, Dv);
}

void ggml_cuda_lod2_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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

    const int l0       = ggml_get_op_params_i32(dst, 0);
    const int s_len    = ggml_get_op_params_i32(dst, 1);
    const int n_routes = ggml_get_op_params_i32(dst, 2);
    const int sink     = ggml_get_op_params_i32(dst, 3);

    const int * q0 = (const int *) meta->data;

    float scale;
    memcpy(&scale, (const char *) dst->op_params + 4*sizeof(int32_t), sizeof(float));

    const int D     = q->ne[0];
    const int nq    = q->ne[1];
    const int Hq    = q->ne[2];
    const int Dv    = v->ne[0];
    const int Hkv   = k->ne[2];
    const int Ds    = s_kv->ne[0];
    const int S_cap = s_kv->ne[1];
    const int P_cap = p_kv->ne[1];
    const int ps    = p_idx->ne[0];
    const int pps   = s_pg->ne[0] - 1;

    cudaStream_t stream = ctx.stream();
    const int wpb = LOD2_BLOCK/LOD2_LANES; // warps per block

    // Generation: one warp per (token, head) would leave the device idle, so the
    // slot logits are computed once and the branches are cut into splits.
    if (nq <= LOD2_DEC_MAX_Q && s_len > 0 && 3*Dv <= LOD2_RED_BLOCK) {
        const int units   = nq*Hq;
        const int protect = std::min(sink, s_len);
        const int R       = std::min(n_routes, std::max(s_len - protect, 0));
        const int n_cu    = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

        // warps in flight = units*NS; the sweep knob multiplies the target
        // Once the s_len/lanes cap was removed, 1 gives NS = 4*n_cu/units = 152
        // splits at 32k and is the optimum: 150.5 t/s against 149.7 / 138.1 /
        // 107.5 for 2 / 4 / 8.  Past that the O(NS) fold in lod2_k_reduce and the
        // per-split fixed work cost more than the extra parallelism buys.
        // Splits are folded LOD2_WPB at a time inside lod2_k_part, so the target
        // is blocks, not warps: aim for four blocks per CU and let the split
        // count follow.  What reaches lod2_k_reduce is NS/LOD2_WPB either way.
        const int nsmul = getenv("LLAMA_LOD2_NSMUL") ? std::max(1, atoi(getenv("LLAMA_LOD2_NSMUL"))) : 1;
        // Each split is one warp of lod2_k_part, so units*NS is its warp count.
        // Measured on 27B/32k generation the kernel is flat once that reaches
        // about ten warps per CU and slower past it, while lod2_k_reduce pays
        // NS/pw linearly - so this aims at the knee rather than at occupancy.
        // The pair (pw, NS) was swept together: 4/128 115.8, 8/128 111.9,
        // 16/128 113.9, 8/202 122.5, 16/202 132.3, 8/404 138.5 us per layer.
        int NS = std::max(1, nsmul*10*n_cu/std::max(units, 1));
        NS = std::min(NS, nsmul*LOD2_WPB*LOD2_DEC_MAX_S);
        // A split may own fewer slots than a lane group - the lanes are spent on
        // the head dimension, not on slots, so a thin split still uses the whole
        // wave.  Capping at s_len/lanes was silently halving the split count when
        // the wave width doubled.
        if (getenv("LLAMA_LOD2_NS")) {
            NS = std::max(1, atoi(getenv("LLAMA_LOD2_NS")));
        }
        NS = std::max(1, std::min(NS, s_len));

        const int g_kv = Hq/Hkv;
        // Query heads per warp.  Sharing the slot read across a GQA group looks
        // like a 4x traffic win and measured 73.7 t/s against 108.5 for one head
        // per warp at 32k: this kernel is occupancy bound, and grouping cuts the
        // warp count by g while inflating the register file.  Kept selectable.
        int g = getenv("LLAMA_LOD2_HPW") ? std::max(1, atoi(getenv("LLAMA_LOD2_HPW"))) : 1;
        // The warp reads one KV head's rows and applies them to G query heads,
        // so G must divide the GQA group - otherwise the heads of a warp would
        // straddle two KV heads and read the wrong means.  g_kv is 6 on this
        // model, which is why the old {1,2,4,8} ladder never had a usable
        // sharing width.
        while (g > 1 && (g_kv % g != 0 || Hq % g != 0)) {
            g--;
        }

        // How many splits one block of lod2_k_part folds before anything
        // reaches memory.  This is what lod2_k_reduce then has to walk, and
        // that walk is exactly linear in it: measured on 27B/32k generation,
        // 8 / 16 / 32 / 51 / 101 partials cost 5.9 / 10.3 / 18.1 / 28.4 /
        // 55.3 us.  The split count itself is flat in lod2_k_part above ~128,
        // so the two are decoupled by folding wider rather than by splitting
        // less: a wider block keeps the same number of waves in flight and
        // hands the reduce a quarter of the partials.
        const int pwm = getenv("LLAMA_LOD2_PWPB") ? std::max(1, atoi(getenv("LLAMA_LOD2_PWPB"))) : 8;
        int pw = 1;
        for (int w = 1; w <= pwm; w *= 2) {
            if (w*LOD2_LANES <= 1024 &&
                (size_t)(w*g*Dv + 2*w*g*3)*sizeof(float) <= 64*1024) {
                pw = w;
            }
        }
        const int nsb = (NS + pw - 1)/pw;               // partials per unit

        const float sl2 = scale*LOD2_LOG2E;
        // Branch mask, for attributing this kernel's cost: 1 coarse, 2 local,
        // 4 routed.  Anything but 7 produces wrong output on purpose - it is
        // how the routed branch was found to be 74% of the kernel.
        const int dbr = getenv("LLAMA_LOD2_DBR") ? atoi(getenv("LLAMA_LOD2_DBR")) : 7;

        ggml_cuda_pool_alloc<float> logits(ctx.pool(), (size_t) units*s_len);
        ggml_cuda_pool_alloc<int>   top   (ctx.pool(), (size_t) units*LOD2_MAX_R);
        ggml_cuda_pool_alloc<int>   rsel  (ctx.pool(), (size_t) units*LOD2_MAX_R);
        ggml_cuda_pool_alloc<float> cmax  (ctx.pool(), (size_t) units);
        ggml_cuda_pool_alloc<float> pm    (ctx.pool(), (size_t) units*nsb*3);
        ggml_cuda_pool_alloc<float> pd    (ctx.pool(), (size_t) units*nsb*3);
        ggml_cuda_pool_alloc<float> pacc  (ctx.pool(), (size_t) units*nsb*3*Dv);

        // Widest load that divides the head, so a mean row moves in as few
        // instructions as its size allows.
        const int wk = D % (4*LOD2_LANES) == 0 ? 4 : (D % (2*LOD2_LANES) == 0 ? 2 : 1);
        const int ek = (D + wk*LOD2_LANES - 1)/(wk*LOD2_LANES);

        if (lgt) {
            // the GEMM already produced q . mean_k; only the count bias is left
            lod2_k_bias<<<(int)(((int64_t) units*s_len + LOD2_BLOCK - 1)/LOD2_BLOCK), LOD2_BLOCK, 0, stream>>>(
                    (const float *) s_kv->data, (const float *) lgt->data, logits.get(),
                    S_cap, Ds, D, Dv, nq, Hq, Hq/Hkv, s_len, (int) lgt->ne[0], sl2);
        } else {
#define LOD2_LAUNCH_LOGITS(W, EV) \
            lod2_k_logits_kv<W, EV><<<(int)(((int64_t) nq*Hkv*s_len + wpb - 1)/wpb), LOD2_BLOCK, 0, stream>>>( \
                    (const char *) q->data, q->nb[1], q->nb[2], \
                    (const float *) s_kv->data, (const float *) s_mn->data, logits.get(), \
                    S_cap, Ds, D, Dv, nq, Hq, Hq/Hkv, s_len, sl2)
            switch (wk*100 + ek) {
                case 401: LOD2_LAUNCH_LOGITS(4, 1); break;
                case 402: LOD2_LAUNCH_LOGITS(4, 2); break;
                case 201: LOD2_LAUNCH_LOGITS(2, 1); break;
                case 202: LOD2_LAUNCH_LOGITS(2, 2); break;
                case 101: LOD2_LAUNCH_LOGITS(1, 1); break;
                case 102: LOD2_LAUNCH_LOGITS(1, 2); break;
                case 104: LOD2_LAUNCH_LOGITS(1, 4); break;
                default:  GGML_ABORT("LoD2: unsupported query head size");
            }
#undef LOD2_LAUNCH_LOGITS
        }

        lod2_k_topk<<<units, LOD2_BLOCK, 0, stream>>>(
                logits.get(), top.get(), cmax.get(), nq, Hq, s_len, protect,
                std::max(1, std::min(R, LOD2_MAX_R)));

        lod2_k_route<<<dim3(LOD2_MAX_R, units), LOD2_BLOCK, 0, stream>>>(
                (const char *) q->data, q->nb[1], q->nb[2],
                (const float *) p_kv->data, (const int *) s_pg->data,
                top.get(), rsel.get(),
                P_cap, S_cap, Ds, D, Dv, ps, pps, Hq/Hkv, nq, Hq, sl2);

        // Widest load that divides the value head, so the coarse pass moves one
        // dwordx4 per lane where it used to issue four lane-strided dwords.
        const int wv  = Dv % (4*LOD2_LANES) == 0 ? 4 : (Dv % (2*LOD2_LANES) == 0 ? 2 : 1);
        const int ev  = (Dv + wv*LOD2_LANES - 1)/(wv*LOD2_LANES);

#define LOD2_LAUNCH_SPLIT(G, W, EV) \
        lod2_k_part<G, W, EV><<<dim3(nsb, nq*(Hq/G)), pw*LOD2_LANES, \
                (pw*(G)*Dv + 2*pw*(G)*3)*sizeof(float), stream>>>( \
                (const char *) q->data, q->nb[1], q->nb[2], \
                (const char *) k->data, lod2_type_code(k->type), k->nb[1], k->nb[2], \
                (const char *) v->data, lod2_type_code(v->type), v->nb[1], v->nb[2], \
                (const float *) s_kv->data, (const float *) s_mn->data, \
                (const float *) p_kv->data, \
                (const int *) p_idx->data, (const int *) s_pg->data, \
                logits.get(), top.get(), cmax.get(), rsel.get(), \
                pm.get(), pd.get(), pacc.get(), \
                S_cap, P_cap, Ds, D, Dv, ps, pps, g_kv, \
                l0, q0, nq, Hq, s_len, NS, sl2, dbr); \
        lod2_k_reduce<<<units, 3*Dv, (3*Dv + 3)*sizeof(float), stream>>>( \
                pm.get(), pd.get(), pacc.get(), \
                (char *) dst->data, dst->nb[1], dst->nb[2], nq, Hq, Dv, nsb)

#define LOD2_LAUNCH_EV(G, W) \
        switch (ev) { \
            case 1: LOD2_LAUNCH_SPLIT(G, W, 1); break; \
            case 2: LOD2_LAUNCH_SPLIT(G, W, 2); break; \
            case 4: LOD2_LAUNCH_SPLIT(G, W, 4); break; \
            case 8: LOD2_LAUNCH_SPLIT(G, W, 8); break; \
            default: GGML_ABORT("LoD2: unsupported value head size"); \
        }

#define LOD2_LAUNCH_W(G) \
        switch (wv) { \
            case 4: LOD2_LAUNCH_EV(G, 4); break; \
            case 2: LOD2_LAUNCH_EV(G, 2); break; \
            default: LOD2_LAUNCH_EV(G, 1); break; \
        }

        switch (g) {
            case 1: LOD2_LAUNCH_W(1); break;
            case 2: LOD2_LAUNCH_W(2); break;
            case 3: LOD2_LAUNCH_W(3); break;
            case 4: LOD2_LAUNCH_W(4); break;
            case 6: LOD2_LAUNCH_W(6); break;
            case 8: LOD2_LAUNCH_W(8); break;
            default: GGML_ABORT("LoD2: unsupported GQA group size");
        }
#undef LOD2_LAUNCH_W
#undef LOD2_LAUNCH_EV
#undef LOD2_LAUNCH_SPLIT
        return;
    }


    // The many-query path always has the GEMM: that is where it pays.  The one
    // exception is a state that is still empty - a prompt shorter than the front
    // window creates no slots, and then a single-token decode falls through the
    // s_len > 0 guard above into here with nothing for the GEMM to have computed.
    // Every consumer of lgt is bounded by s_len, so a null table is safe there;
    // it was the unconditional assert that was wrong.
    GGML_ASSERT(lgt != nullptr || s_len == 0);
    const float * lgt_d = lgt ? (const float *) lgt->data : nullptr;
    const int     lgt_s = lgt ? (int) lgt->ne[0] : 0;
    // This kernel indexes its warps with WARP_SIZE, not LOD2_LANES, so its
    // blocks cover LOD2_BLOCK/WARP_SIZE queries.  Sizing the grid with the
    // 64-wide count launched twice the blocks needed and left half the warps
    // with active == false - and they still walked the routing and coarse
    // branches before reaching the guard, because that guard used to sit after
    // a collective staging step that no longer exists.
    const int wpb_attn = LOD2_BLOCK/WARP_SIZE;

    // Tiled branches, on by default.  Each replaces one of the per-query
    // kernel's three branches with a form that reads a K/V or mean column once
    // per tile of queries instead of once per query; both are verified to give
    // output identical to the per-query path (refinement all-open and default
    // against dense, plus the op oracle) and together roughly double prompt
    // throughput - 27B/32k, 898 -> 1993 t/s on one GPU, 2026 -> 4184 on eight.
    // LLAMA_LOD2_TILED is a bit mask - 1 local, 2 coarse - so either can be
    // measured on its own, and 0 falls back to the per-query kernel, which
    // stays the reference implementation.  Anything that does not fit (shared
    // memory, head sizes, an empty state) falls back on its own below.
    // Rows per block.  Everything these kernels bought came from reuse - a K/V
    // or mean column read once per TM queries instead of once per query - so
    // the row count is the knob that matters and is taken as large as the
    // shared memory allows.  The local kernel pays D*TM floats to stage Q, so
    // it saturates early; the coarse kernel's budget is dominated by the value
    // tile, which does not depend on TM at all, so it can go much wider.
    // K and V already live as F16, so they can be staged narrow with no change
    // to a single arithmetic input.  That, plus folding ps into the K tile's
    // bytes, is what fits this kernel twice into a CU's 64 KB.
    const bool kvh = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;
    const size_t kvw = kvh ? sizeof(half) : sizeof(float);

    auto lds_local  = [&](int tm) {
        const size_t kb  = (size_t) D*LOD2_TN*kvw;
        const size_t psb = (size_t) tm*LOD2_TN*sizeof(float);
        return (size_t) D*tm*sizeof(float) + (kb > psb ? kb : psb) +
               (size_t) LOD2_TN*Dv*kvw;
    };
    auto lds_coarse = [&](int tm) {
        return ((size_t) LOD2_TN*Dv + (size_t) tm*LOD2_TN + LOD2_TN)*sizeof(float);
    };
    // The dimension-split form stages the same two tiles and nothing else, so
    // its footprint is independent of the row count - only of the column tile
    // and of whether the tile is kept narrow.  A wide f32 tile and a narrow f16
    // one cost exactly the same shared memory; the f32 one costs no conversions.
    const int ds_ct  = getenv("LLAMA_LOD2_CT")  ? std::max(1, atoi(getenv("LLAMA_LOD2_CT")))  : 8;
    const int ds_kvh = getenv("LLAMA_LOD2_KVH") ? atoi(getenv("LLAMA_LOD2_KVH")) : 0;
    const bool ds_h  = ds_kvh && kvh;
    const size_t ds_w   = ds_h ? sizeof(half) : sizeof(float);
    const size_t lds_ds = (size_t) ds_ct*(D + Dv)*ds_w;

    // Measured on 27B/32k, coarse branch alone: 16 -> 1014.6, 32 -> 1026.6,
    // 64 -> 1026.0 t/s.  Reuse is saturated by 32, so take the narrower block.
    // The local kernel also stages Q, and D*32 floats of it no longer fit
    // beside the K and value tiles, so 16 is all that is reachable there.
    const int lmax = getenv("LLAMA_LOD2_LM") ? atoi(getenv("LLAMA_LOD2_LM")) : 64;
    const int cmax = getenv("LLAMA_LOD2_CM") ? atoi(getenv("LLAMA_LOD2_CM")) : 32;

    // The dimension-split kernel keeps DP elements of Q and DP accumulators per
    // lane in registers, so DP has to be a compile-time constant and D has to
    // equal Dv.  Anything else falls back to the (query, column) kernel.
    const int  ds_dp   = D/LOD2_TN;
    const int  ds_mode = getenv("LLAMA_LOD2_LOCALDS") ? atoi(getenv("LLAMA_LOD2_LOCALDS")) : 1;
    const bool ds_ok   = ds_mode && D == Dv && D % LOD2_TN == 0 &&
                         (ds_dp == 4 || ds_dp == 8 || ds_dp == 16) && ds_dp % LOD2_CH == 0 &&
                         (ds_ct == 8 || ds_ct == 16) && lds_ds <= 64*1024;
    // Queries per thread.  The row group is fixed at 16 - measured on 27B/32k,
    // local branch alone, RG 32 is 29% slower and RG 64 spills, and the tile no
    // longer depends on the block width, so a wide block buys nothing and costs
    // registers.  RM is the reuse knob; see the kernel.
    const int ds_rm = getenv("LLAMA_LOD2_RM") ? std::max(1, atoi(getenv("LLAMA_LOD2_RM"))) : 2;

    int tm_l = 16, tm_c = 16;
    for (int tm = 16; tm <= 64; tm *= 2) {
        if (tm <= lmax && tm*LOD2_TN <= 1024 && (ds_ok || lds_local(tm) <= 64*1024)) tm_l = tm;
        if (tm <= cmax && tm*LOD2_TN <= 1024 && lds_coarse(tm) <= 64*1024) tm_c = tm;
    }

    const size_t lds_l = ds_ok ? lds_ds : lds_local(tm_l);
    const size_t lds_c = lds_coarse(tm_c);

    const int  tmask = getenv("LLAMA_LOD2_TILED") ? atoi(getenv("LLAMA_LOD2_TILED")) : 3;
    const bool tfits = Dv % LOD2_TN == 0 && Dv/LOD2_TN <= LOD2_TACC && D % 2 == 0;
    const bool tiled_l = (tmask & 1) && tfits && lds_l <= 64*1024;
    // The coarse kernel takes value dimensions in pairs, so it needs Dv to be a
    // multiple of twice the lane count; anything else keeps the per-query branch
    // inside lod2_k_attn, which is the verified fallback either way.
    const int  c_nacc  = Dv/LOD2_TN;
    const bool c_fits  = tfits && Dv % (2*LOD2_TN) == 0 &&
                         (c_nacc == 4 || c_nacc == 8 || c_nacc == 16);
    const bool tiled_c = (tmask & 2) && c_fits && lds_c <= 64*1024 && s_len > 0;

    ggml_cuda_pool_alloc<float> lm  (ctx.pool(), tiled_l ? (size_t) nq*Hq    : 0);
    ggml_cuda_pool_alloc<float> ld  (ctx.pool(), tiled_l ? (size_t) nq*Hq    : 0);
    ggml_cuda_pool_alloc<float> lacc(ctx.pool(), tiled_l ? (size_t) nq*Hq*Dv : 0);
    ggml_cuda_pool_alloc<float> cm  (ctx.pool(), tiled_c ? (size_t) nq*Hq    : 0);
    ggml_cuda_pool_alloc<float> cd  (ctx.pool(), tiled_c ? (size_t) nq*Hq    : 0);
    ggml_cuda_pool_alloc<float> cacc(ctx.pool(), tiled_c ? (size_t) nq*Hq*Dv : 0);

    if (tiled_l && ds_ok) {
#define LOD2_LAUNCH_DS(RG, RM, CTV, DPV, KVH) \
        lod2_k_local_ds<RG, RM, CTV, DPV, KVH><<<dim3((nq + (RG)*(RM) - 1)/((RG)*(RM)), Hq), \
                (RG)*LOD2_TN, lds_ds, stream>>>( \
                (const char *) q->data, q->nb[1], q->nb[2], \
                (const char *) k->data, lod2_type_code(k->type), k->nb[1], k->nb[2], \
                (const char *) v->data, lod2_type_code(v->type), v->nb[1], v->nb[2], \
                lm.get(), ld.get(), lacc.get(), \
                D, Dv, Hq/Hkv, l0, q0, nq, Hq, scale*LOD2_LOG2E)
#define LOD2_LAUNCH_DS_DP(RG, RM, CTV, KVH) \
        switch (ds_dp) { \
            case 16: LOD2_LAUNCH_DS(RG, RM, CTV, 16, KVH); break; \
            case  8: LOD2_LAUNCH_DS(RG, RM, CTV,  8, KVH); break; \
            default: LOD2_LAUNCH_DS(RG, RM, CTV,  4, KVH); break; \
        }
#define LOD2_LAUNCH_DS_CT(RG, RM, KVH) \
        switch (ds_ct) { \
            case 16: LOD2_LAUNCH_DS_DP(RG, RM, 16, KVH); break; \
            default: LOD2_LAUNCH_DS_DP(RG, RM,  8, KVH); break; \
        }
#define LOD2_LAUNCH_DS_RM(RG, KVH) \
        switch (ds_rm) { \
            case 4:  LOD2_LAUNCH_DS_CT(RG, 4, KVH); break; \
            case 2:  LOD2_LAUNCH_DS_CT(RG, 2, KVH); break; \
            default: LOD2_LAUNCH_DS_CT(RG, 1, KVH); break; \
        }
        if (ds_h) {
            LOD2_LAUNCH_DS_RM(16, true)
        } else {
            LOD2_LAUNCH_DS_RM(16, false)
        }
#undef LOD2_LAUNCH_DS_RM
#undef LOD2_LAUNCH_DS_CT
#undef LOD2_LAUNCH_DS_DP
#undef LOD2_LAUNCH_DS
    } else if (tiled_l) {
#define LOD2_LAUNCH_LOCAL(TM, KVH) \
        lod2_k_local<TM, KVH><<<dim3((nq + (TM) - 1)/(TM), Hq), (TM)*LOD2_TN, lds_l, stream>>>( \
                (const char *) q->data, q->nb[1], q->nb[2], \
                (const char *) k->data, lod2_type_code(k->type), k->nb[1], k->nb[2], \
                (const char *) v->data, lod2_type_code(v->type), v->nb[1], v->nb[2], \
                lm.get(), ld.get(), lacc.get(), \
                D, Dv, Hq/Hkv, l0, q0, nq, Hq, scale*LOD2_LOG2E)
#define LOD2_LAUNCH_LOCAL_TM(KVH) \
        switch (tm_l) { \
            case 64: LOD2_LAUNCH_LOCAL(64, KVH); break; \
            case 32: LOD2_LAUNCH_LOCAL(32, KVH); break; \
            default: LOD2_LAUNCH_LOCAL(16, KVH); break; \
        }
        if (kvh) {
            LOD2_LAUNCH_LOCAL_TM(true)
        } else {
            LOD2_LAUNCH_LOCAL_TM(false)
        }
#undef LOD2_LAUNCH_LOCAL_TM
#undef LOD2_LAUNCH_LOCAL
    }

    if (tiled_c) {
#define LOD2_LAUNCH_COARSE(TM, NA) \
        lod2_k_coarse<TM, NA><<<dim3((nq + (TM) - 1)/(TM), Hq), (TM)*LOD2_TN, lds_c, stream>>>( \
                (const float *) s_kv->data, (const float *) s_mn->data, \
                lgt_d, \
                cm.get(), cd.get(), cacc.get(), \
                S_cap, Ds, D, Dv, Hq/Hkv, nq, s_len, lgt_s, \
                n_routes, sink, scale*LOD2_LOG2E)
#define LOD2_LAUNCH_COARSE_NA(TM) \
        switch (c_nacc) { \
            case 16: LOD2_LAUNCH_COARSE(TM, 16); break; \
            case  8: LOD2_LAUNCH_COARSE(TM,  8); break; \
            default: LOD2_LAUNCH_COARSE(TM,  4); break; \
        }
        switch (tm_c) {
            case 64: LOD2_LAUNCH_COARSE_NA(64); break;
            case 32: LOD2_LAUNCH_COARSE_NA(32); break;
            default: LOD2_LAUNCH_COARSE_NA(16); break;
        }
#undef LOD2_LAUNCH_COARSE_NA
#undef LOD2_LAUNCH_COARSE
    }

    lod2_k_attn<<<dim3((nq + wpb_attn - 1)/wpb_attn, Hq), LOD2_BLOCK, 0, stream>>>(
            (const char *) q->data, q->nb[1], q->nb[2],
            (const char *) k->data, lod2_type_code(k->type), k->nb[1], k->nb[2],
            (const char *) v->data, lod2_type_code(v->type), v->nb[1], v->nb[2],
            (const float *) s_kv->data, (const float *) s_mn->data,
            lgt_d,
            (const float *) p_kv->data,
            (const int *) p_idx->data, (const int *) s_pg->data,
            (char *) dst->data, dst->nb[1], dst->nb[2],
            S_cap, P_cap, Ds, D, Dv, ps, pps, Hq/Hkv,
            l0, q0, nq, s_len, lgt_s, n_routes, sink, scale,
            tiled_l ? lm.get() : nullptr, tiled_l ? ld.get() : nullptr,
            tiled_l ? lacc.get() : nullptr,
            tiled_c ? cm.get() : nullptr, tiled_c ? cd.get() : nullptr,
            tiled_c ? cacc.get() : nullptr);
}
