// test-lod2.cpp: replay a LoD2 oracle case through ggml_lod2_update / ggml_lod2_attn.
//
// The case files are produced by docs/lod2-pytorch/dump_lod2_case.py from the
// standalone PyTorch reference, which is itself checked against the author's
// implementation by docs/lod2-pytorch/check_lod2_ref.py.  So the chain is
//
//   author's LOD attention  ==  lod2_ref.py  ==  these ops
//
// with the first link verified in Python and the second one here.  The
// schedule comes from src/llama-lod2.h, the same header the graph builder uses,
// so a schedule change cannot silently diverge between the two.
//
//   ./bin/test-lod2 <case.bin> [<case.bin> ...]
//
// With no arguments the test looks for the cases the CI script generates and
// skips (exit 0) when none are present, so it is harmless in a stock checkout.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-lod2.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct lod2_case {
    int32_t n_head_q, n_head_kv, head_dim, value_dim;
    int32_t n_prompt, n_decode;
    float   scale;
    int32_t chunk_len, local_len, sink_len, page_size, pages_per_slot;
    int32_t routes_prefill, routes_decode;
    int32_t prefill_chunk, prefill_local;
    int32_t prefill_state_update, decode_state_update;
    float   state_growth;
    int32_t state_min;
    int32_t s_cap, p_cap, state_len, coverage;

    std::vector<float> q, k, v, out;          // [T][H][D]
    std::vector<float> key_sum, value_sum, count;

    int32_t total() const { return n_prompt + n_decode; }

    llama_lod2_params params() const {
        llama_lod2_params p;
        p.chunk_len            = chunk_len;
        p.local_len            = local_len;
        p.sink_len             = sink_len;
        p.page_size            = page_size;
        p.pages_per_slot       = pages_per_slot;
        p.state_growth         = state_growth;
        p.state_min            = state_min;
        p.routes_prefill       = routes_prefill;
        p.routes_decode        = routes_decode;
        p.prefill_chunk        = prefill_chunk;
        p.prefill_local        = prefill_local;
        p.prefill_state_update = prefill_state_update;
        p.decode_state_update  = decode_state_update;
        return p;
    }
};

static bool read_exact(FILE * fh, void * dst, size_t n) {
    return fread(dst, 1, n, fh) == n;
}

static bool load_case(const char * path, lod2_case & c) {
    FILE * fh = fopen(path, "rb");
    if (!fh) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    char magic[8];
    int32_t version = 0;
    bool ok = read_exact(fh, magic, 8) && memcmp(magic, "LOD2CASE", 8) == 0 &&
              read_exact(fh, &version, 4) && version == 1;
    if (!ok) {
        fprintf(stderr, "%s: not a version-1 LOD2CASE file\n", path);
        fclose(fh);
        return false;
    }
    ok = ok && read_exact(fh, &c.n_head_q, 4*4);
    ok = ok && read_exact(fh, &c.n_prompt, 2*4);
    ok = ok && read_exact(fh, &c.scale, 4);
    ok = ok && read_exact(fh, &c.chunk_len, 5*4);
    ok = ok && read_exact(fh, &c.routes_prefill, 2*4);
    ok = ok && read_exact(fh, &c.prefill_chunk, 2*4);
    ok = ok && read_exact(fh, &c.prefill_state_update, 2*4);
    ok = ok && read_exact(fh, &c.state_growth, 4);
    ok = ok && read_exact(fh, &c.state_min, 4);
    ok = ok && read_exact(fh, &c.s_cap, 4*4);
    if (!ok) {
        fprintf(stderr, "%s: truncated header\n", path);
        fclose(fh);
        return false;
    }

    const size_t T = c.total();
    const auto load = [&](std::vector<float> & dst, size_t n) {
        dst.resize(n);
        ok = ok && read_exact(fh, dst.data(), n*sizeof(float));
    };
    load(c.q,   T*c.n_head_q *c.head_dim);
    load(c.k,   T*c.n_head_kv*c.head_dim);
    load(c.v,   T*c.n_head_kv*c.value_dim);
    load(c.out, T*c.n_head_q *c.value_dim);
    load(c.key_sum,   (size_t) c.n_head_kv*c.s_cap*c.head_dim);
    load(c.value_sum, (size_t) c.n_head_kv*c.s_cap*c.value_dim);
    load(c.count,     (size_t) c.n_head_kv*c.s_cap);
    fclose(fh);
    if (!ok) {
        fprintf(stderr, "%s: truncated payload\n", path);
        return false;
    }
    return true;
}

// [T][H][D] (case order) -> [D, T, H] (op order)
static std::vector<float> to_op_order(const std::vector<float> & src, int64_t T, int64_t H, int64_t D) {
    std::vector<float> dst((size_t) T*H*D);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            memcpy(&dst[(size_t) (h*T + t)*D], &src[(size_t) (t*H + h)*D], D*sizeof(float));
        }
    }
    return dst;
}

struct diff {
    double max_abs = 0.0;
    double sum_abs = 0.0;
    size_t n       = 0;

    void add(double a, double b) {
        const double d = std::fabs(a - b);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        n++;
    }
    double mean_abs() const { return n ? sum_abs/n : 0.0; }
};

static bool run_case(const char * path, ggml_backend_t backend, const char * bname) {
    lod2_case c;
    if (!load_case(path, c)) {
        return false;
    }
    const llama_lod2_params p = c.params();

    const int64_t T   = c.total();
    const int64_t D   = c.head_dim;
    const int64_t Dv  = c.value_dim;
    const int64_t Hq  = c.n_head_q;
    const int64_t Hkv = c.n_head_kv;
    const int64_t Ds  = D + Dv + 1;

    printf("%s [%s]\n  prompt %d decode %d  heads %" PRId64 "/%" PRId64 " dim %" PRId64
           "  s_cap %d p_cap %d\n",
           path, bname, c.n_prompt, c.n_decode, Hq, Hkv, D, c.s_cap, c.p_cap);

    if ((uint32_t) c.s_cap != p.slot_capacity(c.n_prompt) ||
        (uint32_t) c.p_cap != p.page_capacity(c.n_prompt)) {
        fprintf(stderr, "  capacity mismatch: header says %d/%d, schedule says %u/%u\n",
                c.s_cap, c.p_cap, p.slot_capacity(c.n_prompt), p.page_capacity(c.n_prompt));
        return false;
    }

    // persistent tensors live in their own buffer: the ops write into them and
    // they must survive across graphs
    ggml_init_params sp = { ggml_tensor_overhead()*4096, nullptr, true };
    ggml_context * sctx = ggml_init(sp);

    ggml_tensor * k     = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, D,  T, Hkv);
    ggml_tensor * v     = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, Dv, T, Hkv);
    ggml_tensor * qt    = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, D,  T, Hq);
    ggml_tensor * s_kv  = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, Ds, c.s_cap, Hkv);
    ggml_tensor * s_mn  = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, Ds - 1, c.s_cap, Hkv);
    ggml_tensor * p_kv  = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, Ds, c.p_cap, Hkv);
    ggml_tensor * p_idx = ggml_new_tensor_3d(sctx, GGML_TYPE_I32, p.page_size, c.p_cap, Hkv);
    ggml_tensor * s_pg  = ggml_new_tensor_3d(sctx, GGML_TYPE_I32, p.pages_per_slot + 1, c.s_cap, Hkv);
    ggml_tensor * meta  = ggml_new_tensor_2d(sctx, GGML_TYPE_I32, 4, Hkv);
    // pool of per-node q0 tensors: they must exist before the context is
    // allocated, and one replay uses a handful
    std::vector<ggml_tensor *> meta_pool;
    for (int i = 0; i < 1024; ++i) {
        meta_pool.push_back(ggml_new_tensor_1d(sctx, GGML_TYPE_I32, 4));
    }

    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);

    {
        const std::vector<float> kd = to_op_order(c.k, T, Hkv, D);
        const std::vector<float> vd = to_op_order(c.v, T, Hkv, Dv);
        const std::vector<float> qd = to_op_order(c.q, T, Hq,  D);
        ggml_backend_tensor_set(k,  kd.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v,  vd.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(qt, qd.data(), 0, ggml_nbytes(qt));

        const std::vector<float>   zf((size_t) std::max(ggml_nelements(s_kv), ggml_nelements(p_kv)), 0.0f);
        const std::vector<int32_t> zi((size_t) std::max(ggml_nelements(p_idx), ggml_nelements(s_pg)), 0);
        const std::vector<int32_t> mi((size_t) ggml_nelements(p_idx), -1);
        ggml_backend_tensor_set(s_kv,  zf.data(), 0, ggml_nbytes(s_kv));
        const std::vector<uint8_t> zm((size_t) ggml_nbytes(s_mn), 0);
        ggml_backend_tensor_set(s_mn,  zm.data(), 0, ggml_nbytes(s_mn));
        ggml_backend_tensor_set(p_kv,  zf.data(), 0, ggml_nbytes(p_kv));
        ggml_backend_tensor_set(p_idx, mi.data(), 0, ggml_nbytes(p_idx));
        ggml_backend_tensor_set(s_pg,  zi.data(), 0, ggml_nbytes(s_pg));
        ggml_backend_tensor_set(meta,  zi.data(), 0, ggml_nbytes(meta));
    }

    // ---- replay -----------------------------------------------------------
    const size_t gsize = ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096, false);
    ggml_init_params gp = { gsize, nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 4096, false);

    ggml_tensor * cur_state = s_kv;
    std::vector<ggml_tensor *> outs;
    std::vector<int32_t> q0_of;

    const auto add_updates = [&](const std::vector<llama_lod2_block> & blocks) {
        for (const auto & b : blocks) {
            cur_state = ggml_lod2_update(gctx, k, v, cur_state, s_mn, p_kv, p_idx, s_pg, meta,
                    b.p0, b.p1, b.s_len, b.s_len_new, p.sink_len);
            ggml_build_forward_expand(gf, cur_state);
        }
    };
    // one meta tensor per attention node: the replay builds the whole schedule
    // as a single graph, so each node needs its own q0
    std::vector<ggml_tensor *> metas;
    const auto add_attn = [&](int64_t q0, int64_t nq, uint32_t l0, uint32_t s_len, uint32_t routes) {
        ggml_tensor * qv = ggml_view_3d(gctx, qt, D, nq, Hq,
                qt->nb[1], qt->nb[2], q0*qt->nb[1]);
        GGML_ASSERT(metas.size() < meta_pool.size());
        ggml_tensor * am = meta_pool[metas.size()];
        metas.push_back(am);
        q0_of.push_back((int32_t) q0);
        ggml_tensor * mk = ggml_view_3d(gctx, s_mn, D, std::max<uint32_t>(s_len, 1), Hkv,
                s_mn->nb[1], s_mn->nb[2], 0);
        ggml_tensor * lg = ggml_mul_mat(gctx, ggml_cont(gctx, mk), qv);
        ggml_tensor * o = ggml_lod2_attn(gctx, qv, k, v, cur_state, s_mn, p_kv, p_idx, s_pg,
                am, lg, l0, s_len, routes, p.sink_len, c.scale);
        ggml_build_forward_expand(gf, o);
        outs.push_back(o);
    };

    uint32_t coverage = 0;
    uint32_t s_len    = 0;

    const uint32_t front = std::min<uint32_t>(c.n_prompt, p.front_len());
    add_attn(0, front, 0, 0, 0);

    for (uint32_t qb = front; qb < (uint32_t) c.n_prompt; qb += p.prefill_chunk) {
        const uint32_t qe = std::min<uint32_t>(c.n_prompt, qb + p.prefill_chunk);
        const uint32_t lb = p.local_begin(qb);

        const auto blocks = llama_lod2_plan_prefill(p, coverage, s_len, lb);
        add_updates(blocks);
        if (!blocks.empty()) {
            coverage = blocks.back().p1;
            s_len    = blocks.back().s_len_new;
        }
        if (coverage != lb) {
            fprintf(stderr, "  coverage drifted: %u != %u\n", coverage, lb);
            return false;
        }
        add_attn(qb, qe - qb, lb, s_len, p.routes_prefill);
    }

    {
        const uint32_t target = p.decode_begin(c.n_prompt + 1);
        const auto blocks = llama_lod2_plan_tail(
                p, coverage, s_len, target, c.n_prompt, p.prefill_state_update);
        add_updates(blocks);
        if (!blocks.empty()) {
            coverage = blocks.back().p1;
            s_len    = blocks.back().s_len_new;
        }
    }

    for (int32_t step = 0; step < c.n_decode; ++step) {
        const uint32_t total  = c.n_prompt + step + 1;
        const uint32_t target = p.decode_begin(total);
        const auto blocks = llama_lod2_plan_tail(
                p, coverage, s_len, target, total, p.decode_state_update);
        add_updates(blocks);
        if (!blocks.empty()) {
            coverage = blocks.back().p1;
            s_len    = blocks.back().s_len_new;
        }
        add_attn(total - 1, 1, coverage, s_len, p.routes_decode);
    }

    if (s_len != (uint32_t) c.state_len || coverage != (uint32_t) c.coverage) {
        fprintf(stderr, "  final state_len/coverage %u/%u != oracle %d/%d\n",
                s_len, coverage, c.state_len, c.coverage);
        return false;
    }

    // the meta tensors live in the persistent context, so they are filled here
    for (size_t i = 0; i < metas.size(); ++i) {
        const int32_t m[4] = { q0_of[i], 0, 0, 0 };
        ggml_backend_tensor_set(metas[i], m, 0, sizeof(m));
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "  graph allocation failed\n");
        return false;
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  graph compute failed\n");
        return false;
    }

    // ---- compare -----------------------------------------------------------
    diff dout;
    int64_t row = 0;
    for (ggml_tensor * o : outs) {
        const int64_t n = o->ne[2];
        std::vector<float> got((size_t) ggml_nelements(o));
        ggml_backend_tensor_get(o, got.data(), 0, ggml_nbytes(o));
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t h = 0; h < Hq; ++h) {
                for (int64_t d = 0; d < Dv; ++d) {
                    dout.add(got[(size_t) (t*Hq + h)*Dv + d],
                             c.out[(size_t) ((row + t)*Hq + h)*Dv + d]);
                }
            }
        }
        row += n;
    }

    std::vector<float> st((size_t) ggml_nelements(s_kv));
    ggml_backend_tensor_get(s_kv, st.data(), 0, ggml_nbytes(s_kv));
    diff dk, dv, dc;
    for (int64_t h = 0; h < Hkv; ++h) {
        for (int64_t s = 0; s < c.state_len; ++s) {
            const float * r = &st[(size_t) (h*c.s_cap + s)*Ds];
            for (int64_t i = 0; i < D; ++i) {
                dk.add(r[i], c.key_sum[(size_t) (h*c.s_cap + s)*D + i]);
            }
            for (int64_t i = 0; i < Dv; ++i) {
                dv.add(r[D + i], c.value_sum[(size_t) (h*c.s_cap + s)*Dv + i]);
            }
            dc.add(r[D + Dv], c.count[(size_t) h*c.s_cap + s]);
        }
    }

    printf("  output     max_abs %.3e  mean_abs %.3e  (%zu values)\n",
            dout.max_abs, dout.mean_abs(), dout.n);
    printf("  key_sum    max_abs %.3e   value_sum max_abs %.3e   count max_abs %.3e\n",
            dk.max_abs, dv.max_abs, dc.max_abs);

    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    ggml_free(sctx);
    ggml_backend_buffer_free(sbuf);

    // The reference accumulates the state in float32 in a different order, so
    // the sums differ by rounding; the counts are integers and must be exact.
    const bool ok = dout.max_abs < 2e-3 && dc.max_abs == 0.0 && dk.max_abs < 1e-2;
    printf("  %s\n", ok ? "OK" : "FAILED");
    return ok;
}

int main(int argc, char ** argv) {
    std::vector<std::string> cases;
    for (int i = 1; i < argc; ++i) {
        cases.push_back(argv[i]);
    }
    if (cases.empty()) {
        const char * env = getenv("LLAMA_LOD2_CASES");
        if (env) {
            cases.push_back(env);
        }
    }
    if (cases.empty()) {
        printf("test-lod2: no case files given, skipping\n"
               "  generate one with docs/lod2-pytorch/dump_lod2_case.py\n");
        return 0;
    }
    // every backend that claims the ops runs the same cases: the CPU path is
    // the contract, so a backend that disagrees with it is the one that is wrong
    bool ok = true;
    bool any = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "failed to init %s\n", ggml_backend_dev_name(dev));
            continue;
        }
        any = true;
        for (const auto & path : cases) {
            ok = run_case(path.c_str(), backend, ggml_backend_dev_name(dev)) && ok;
        }
        ggml_backend_free(backend);
    }
    if (!any) {
        fprintf(stderr, "no usable backend\n");
        return 1;
    }
    return ok ? 0 : 1;
}
