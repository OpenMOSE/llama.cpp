// llama-lod-search - derive per-layer LoD page budgets from one calibration pass.
//
// The expensive thing about tuning per-layer budgets is that a naive search needs one
// model run per candidate configuration. It does not have to: for a given query, the
// quantity that decides whether a budget is enough is how much of the dense attention
// mass lands in the pages the LoD selector would pick. Both halves of that - the true
// per-page mass and the selector's per-page score - come out of a SINGLE dense forward
// pass (LLAMA_LOD_PROFILE=1 emits them per layer). Ranking pages by score and taking the
// cumulative mass then gives capture(k) for every k at once, so the search collapses to
// reading a curve.
//
//   llama-lod-search -m model.gguf -f calibration.txt -c 32768 \
//                    --target-capture 0.95 --range 8:512 -o lod-config.json

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct layer_profile {
    std::vector<float> mass;   // [n_pages] dense attention mass per page (summed over heads)
    std::vector<float> score;  // [n_pages] LoD selection score per page (max over heads)
    std::vector<float> qscore; // [n_tokens, n_pages] head-pooled score of every query
    int64_t            n_tok = 0;
};

struct profile_collector {
    std::map<int, layer_profile> layers;

    static bool cb(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * self = (profile_collector *) user_data;

        const char * name = t->name;
        const bool is_mass  = strncmp(name, "lod_prof_mass",   13) == 0;
        const bool is_score = strncmp(name, "lod_prof_score",  14) == 0;
        const bool is_qsc   = strncmp(name, "lod_prof_qscore", 15) == 0;

        if (ask) {
            return is_mass || is_score || is_qsc;
        }
        if (!is_mass && !is_score && !is_qsc) {
            return true;
        }

        // names are "<tag>-<il>"
        const char * dash = strrchr(name, '-');
        if (!dash) {
            return true;
        }
        const int il = atoi(dash + 1);

        const int64_t n_pages = t->ne[0];
        const int64_t n_head  = t->ne[1];

        std::vector<float> buf(n_pages*n_head);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size()*sizeof(float));

        if (is_qsc) {
            // [n_tokens, n_pages]
            self->layers[il].qscore = std::move(buf);
            self->layers[il].n_tok  = n_pages; // ne[0] is n_tokens here
            return true;
        }

        std::vector<float> agg(n_pages, is_mass ? 0.0f : -INFINITY);
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t p = 0; p < n_pages; ++p) {
                const float v = buf[p + h*n_pages];
                if (is_mass) {
                    agg[p] += v;                       // total mass over heads
                } else {
                    agg[p] = std::max(agg[p], v);      // the selector max-pools over heads
                }
            }
        }
        if (is_mass) {
            const float inv = (float) n_head;
            for (auto & v : agg) {
                v /= inv;                              // mean mass per head: sums to 1
            }
            self->layers[il].mass = std::move(agg);
        } else {
            self->layers[il].score = std::move(agg);
        }
        return true;
    }
};

// cumulative dense mass captured by the top-k pages of the selector's ranking
static std::vector<float> capture_curve(const layer_profile & lp) {
    const size_t n = lp.mass.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return lp.score[a] > lp.score[b] || (lp.score[a] == lp.score[b] && a < b);
    });

    std::vector<float> cum(n + 1, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        cum[i + 1] = cum[i] + lp.mass[idx[i]];
    }
    const float total = cum[n] > 0.0f ? cum[n] : 1.0f;
    for (auto & v : cum) {
        v /= total;
    }
    return cum; // cum[k] = fraction of mass captured by k pages
}

static void print_usage(int, char **) {
    LOG("\nusage: llama-lod-search -m MODEL -f CALIBRATION [options]\n\n");
    LOG("  --target WHAT        what to hold fixed while the other side is optimised:\n");
    LOG("                         cost=N      spend N pages in total, maximise the worst layer\n");
    LOG("                         cost=8%%     same, expressed as the share of context read\n");
    LOG("                                     exactly (pages per layer / total pages)\n");
    LOG("                         quality=F   reach capture F in every layer, minimise cost\n");
    LOG("                         pareto      print the cost/quality frontier and stop\n");
    LOG("                       Cost is linear in pages for both prefill and decode, so pages\n");
    LOG("                       are a faithful stand-in for time; quality is the fraction of\n");
    LOG("                       dense attention mass the selected pages carry.\n");
    LOG("                       If you are unsure which to pin, run --target pareto first.\n");
    LOG("  --probes FILE        score retrieval as a log-probability instead of profiling:\n");
    LOG("                       teacher-force each probe's prefix and read the log-prob of\n");
    LOG("                       the correct answer. Continuous and deterministic, unlike a\n");
    LOG("                       pass/fail needle score which moves +-2 of 12 on FP noise.\n");
    LOG("  --eval-top SPEC      budget to score in --probes mode (same grammar as\n");
    LOG("                       --lod-top-pages); omit to score whatever the env selects\n");
    LOG("  --union              also report how many pages the per-32-query-block selections\n");
    LOG("                       need in TOTAL. If that union is small, per-block selection\n");
    LOG("                       could be served by a compact gather instead of mask's\n");
    LOG("                       full-span read; if it approaches the page count, it cannot.\n");
    LOG("  --granularity G      ubatch (default) profiles one page set shared by the whole\n");
    LOG("                       ubatch, which is what --lod-prefill gather does; block\n");
    LOG("                       profiles a single 32-query block, i.e. what mask mode gets.\n");
    LOG("                       Comparing the two frontiers is how you price the granularity.\n");
    LOG("  --min-lift F         (no --target) buy pages while worth F times more than a\n");
    LOG("                       random page (default 2.0)\n");
    LOG("  --target-capture F   cap for the --min-lift mode (default 0.95)\n");
    LOG("  --range MIN:MAX      clamp the per-layer budget (default 8:512). The FLOOR is a\n");
    LOG("                       real trade: 8 lets the search find peaks but starves quiet\n");
    LOG("                       layers (coverage drops); 3/4 of the baseline protects them\n");
    LOG("                       but takes the pages from the peaks (needle drops). Measure\n");
    LOG("                       both - on gemma4 they tie, on qwen3.6 they do not\n");
    LOG("  --probes FILE        score retrieval log-probability with these probes\n");
    LOG("  --niah F=V[,F=V...]  score the END-TO-END needle benchmark instead: prefill each\n");
    LOG("                       document, generate greedily, count how many contain V.\n");
    LOG("                       This is the objective, not a proxy for it - slower, and the\n");
    LOG("                       only one that has not picked a losing allocation here\n");
    LOG("  --niah-n N           tokens to generate per needle document (default 12)\n");

    LOG("  --search MODE        sens   - per-layer sensitivity only\n");
    LOG("                       greedy - redistribute a fixed total by sensitivity\n");
    LOG("                       trim   - take pages off the layers that do not miss them,\n");
    LOG("                                keeping retrieval within --tie of the start\n");
    LOG("  --phase prefill|decode  which budget the search moves (default prefill)\n");
    LOG("  --eval-top N         the uniform budget the search starts from (default 32)\n");
    LOG("  --max-eval N         stop the search after N model evaluations (default 60)\n");
    LOG("  --guard N            hold out the last N document tokens as a language-modelling\n");
    LOG("                       check, so a candidate cannot win by wrecking everything but\n");
    LOG("                       retrieval (default 256, 0 disables)\n");
    LOG("  --guard-tol NATS     reject a candidate that loses more than this much on the\n");
    LOG("                       guard (default 0.05)\n");
    LOG("  --tie NATS           treat retrieval differences below this as ties and settle\n");
    LOG("                       them on the guard instead (default 0.05)\n");
    LOG("  --round N            budget step, also the granularity of the knee (default 8)\n");
    LOG("  -o, --out PATH       write the JSON config here (default lod-config.json)\n");
    LOG("\nThe calibration prompt should be at least as long as the contexts you care about:\n");
    LOG("capture is measured over the complete pages that exist when the last token is read.\n\n");
}

int main(int argc, char ** argv) {
    common_params params;

    float       target  = 0.95f;
    uint32_t    kmin    = 8;
    bool        range_set = false;
    uint32_t    kmax    = 512;
    uint32_t    round_to = 8;
    float       min_lift = 2.0f;
    int         budget_total = -1;
    std::string mode;            // "", "cost", "quality", "pareto"
    float       quality_floor = 0.0f;
    std::string cost_arg;
    bool        granularity_block = false;
    bool        want_union = false;
    std::string probes_path;
    std::string eval_top;
    std::string search_mode;
    std::string phase   = "prefill";
    std::string niah_spec;
    int         niah_n = 12;
    int         max_eval = 60;
    int         n_guard  = 128;
    double      guard_tol = 0.05;
    double      tie       = 0.05;
    std::string out     = "lod-config.json";

    // pull our own options out before the common parser sees them
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--target-capture") { const char * v = next(); if (v) target = atof(v); continue; }
        if (a == "--min-lift")       { const char * v = next(); if (v) min_lift = atof(v); continue; }
        if (a == "--union") { want_union = true; continue; }
        if (a == "--probes") { const char * v = next(); if (v) probes_path = v; continue; }
        if (a == "--eval-top") { const char * v = next(); if (v) eval_top = v; continue; }
        if (a == "--search") { const char * v = next(); if (v) search_mode = v; continue; }
        if (a == "--phase")  { const char * v = next(); if (v) phase = v; continue; }
        if (a == "--niah")   { const char * v = next(); if (v) niah_spec = v; continue; }
        if (a == "--niah-n") { const char * v = next(); if (v) niah_n = atoi(v); continue; }
        if (a == "--max-eval") { const char * v = next(); if (v) max_eval = atoi(v); continue; }
        if (a == "--guard") { const char * v = next(); if (v) n_guard = atoi(v); continue; }
        if (a == "--guard-tol") { const char * v = next(); if (v) guard_tol = atof(v); continue; }
        if (a == "--tie") { const char * v = next(); if (v) tie = atof(v); continue; }
        if (a == "--granularity") {
            const char * v = next();
            granularity_block = v && strncmp(v, "block", 5) == 0;
            continue;
        }
        if (a == "--budget-total")   { const char * v = next(); if (v) budget_total = atoi(v); continue; }
        if (a == "--target") {
            const char * v = next();
            if (v) {
                if (strncmp(v, "pareto", 6) == 0) {
                    mode = "pareto";
                } else if (strncmp(v, "quality=", 8) == 0) {
                    mode = "quality";
                    quality_floor = atof(v + 8);
                } else if (strncmp(v, "cost=", 5) == 0) {
                    mode = "cost";
                    cost_arg = v + 5;
                } else {
                    throw std::invalid_argument("--target must be cost=N, cost=P%, quality=F or pareto");
                }
            }
            continue;
        }
        if (a == "--round")          { const char * v = next(); if (v) round_to = std::max(1, atoi(v)); continue; }
        if (a == "-o" || a == "--out") { const char * v = next(); if (v) out = v; continue; }
        if (a == "--range") {
            range_set = true;
            const char * v = next();
            if (v) {
                kmin = atoi(v);
                const char * c = strchr(v, ':');
                if (c) kmax = atoi(c + 1);
            }
            continue;
        }
        rest.push_back(argv[i]);
    }

    common_init();

    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY, print_usage)) {
        return 1;
    }

    // profile the DENSE read: a saturating budget makes the LoD path bit-exact dense, and
    // the profile branch then reports the true attention mass
    setenv("LLAMA_LOD", "1", 1);
    // match what --lod does on the command line. Without this the tool measures the composed
    // decode path, which is not what anyone runs - and which asserts at a 1-token decode.
    setenv("LLAMA_LOD_FUSED", "1", 0);
    // profiling needs the read to BE dense (saturating budget); probe mode is the opposite -
    // it scores whatever configuration the caller wants to judge
    if (probes_path.empty()) {
        setenv("LLAMA_LOD_TOP_PAGES", "1000000", 1);
    } else if (!eval_top.empty()) {
        setenv("LLAMA_LOD_TOP_PAGES", eval_top.c_str(), 1);
    }
    // do not clobber an explicit choice: =1 profiles ubatch-shared selection
    // (gather), =2 profiles one 32-query block (mask granularity)
    setenv("LLAMA_LOD_PROFILE", want_union ? "3" : (granularity_block ? "2" : "1"), 1);
    unsetenv("LLAMA_LOD_TOP_PAGES_DECODE");
    unsetenv("LLAMA_LOD_PREFILL");

    profile_collector collector;
    params.cb_eval           = profile_collector::cb;
    params.cb_eval_user_data = &collector;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * ctx   = llama_init ? llama_init->context() : nullptr;
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }

    // ---- retrieval scored as a log-probability (the objective that must not regress).
    // Pass/fail retrieval moves by +-2 of 12 with nothing but FP-level perturbations, which
    // is more than any budget decision is worth. Teacher-forcing "<key> = " and reading the
    // log-probability of the correct value measures the same thing continuously and
    // deterministically, and it never goes through generation.
    if (!probes_path.empty() || !niah_spec.empty()) {
        nlohmann::json pj = nlohmann::json::array();
        if (!probes_path.empty()) {
            std::ifstream pf(probes_path);
            if (!pf) {
                LOG_ERR("%s: cannot open %s\n", __func__, probes_path.c_str());
                return 1;
            }
            pj = nlohmann::json::parse(pf, nullptr, true, true);
        }

        const bool dec_phase = (phase == "decode");

        const std::vector<llama_token> doc = common_tokenize(ctx, params.prompt, true);
        llama_memory_t mem = llama_get_memory(ctx);

        // Discover which layers LoD is actually wired into. This has to be a full pass over
        // the document: with only a handful of tokens there are no pages to select and the
        // profile branch emits nothing, which would leave the search scanning every layer in
        // the model - most of them ignoring their budget entirely.
        {
            llama_memory_clear(mem, true);
            const uint32_t nb = llama_n_batch(ctx);
            for (size_t i = 0; i < doc.size(); i += nb) {
                const uint32_t n = (uint32_t) std::min<size_t>(nb, doc.size() - i);
                llama_batch b = llama_batch_get_one(const_cast<llama_token *>(doc.data() + i), n);
                if (llama_decode(ctx, b) != 0) {
                    LOG_ERR("%s: layer discovery pass failed\n", __func__);
                    return 1;
                }
            }
            llama_memory_clear(mem, true);
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        // Language-modelling guard: the last `n_guard` tokens of the document are scored as
        // ordinary next-token prediction. Retrieval alone can be satisfied by one layer with
        // a huge budget while every other layer is starved; this says whether the model can
        // still read plain text in that state. It is read out of the SAME prefill the probes
        // run on - holding tokens out instead would change the context the probes see, and
        // with it the answer to the question being asked.
        const size_t n_ctxdoc = doc.size();

        // The guard is a KL divergence against the DENSE read of the same positions, not a
        // plain NLL. A starved model gets more confident on repetitive text, so its NLL can
        // improve while its distribution is badly wrong - measured here: an allocation that
        // gained 1.5 nats of NLL lost 2/12 on coverage. KL cannot be gamed that way.
        double              guard_nll = 0.0;    // mean KL once the reference is captured
        std::vector<float>  dense_lp;           // [n_guard][n_vocab] log-probs of the dense read
        bool                capture_dense = false;

        // ---- the end-to-end needle objective.
        // Every proxy tried here (attention-mass capture, teacher-forced retrieval
        // log-probability, KL against dense) has picked allocations that then LOST on the
        // real benchmark. So the search can also run the benchmark itself: prefill each
        // needle document, generate greedily, and check whether the planted value appears.
        // Coarse (one integer per document) but it is the thing being optimised for.
        struct niah_case { std::vector<llama_token> prompt; std::string want; std::string path; };
        std::vector<niah_case> niah;
        if (!niah_spec.empty()) {
            std::stringstream ss(niah_spec);
            std::string item;
            while (std::getline(ss, item, ',')) {
                const size_t eq = item.rfind('=');
                if (eq == std::string::npos) {
                    LOG_ERR("%s: --niah wants file=value pairs, got '%s'\n", __func__, item.c_str());
                    return 1;
                }
                const std::string path = item.substr(0, eq);
                std::ifstream nf(path);
                if (!nf) {
                    LOG_ERR("%s: cannot open needle document %s\n", __func__, path.c_str());
                    return 1;
                }
                std::stringstream buf;
                buf << nf.rdbuf();
                niah.push_back({ common_tokenize(ctx, buf.str(), true), item.substr(eq + 1), path });
                LOG_INF("needle: %s (%zu tokens) wants \"%s\"\n",
                        path.c_str(), niah.back().prompt.size(), niah.back().want.c_str());
            }
        }
        int niah_pass = 0;   // how many needle documents the last candidate answered

        auto score_niah = [&](llama_context * cx) -> double {
            niah_pass = 0;
            double margin = 0.0;      // continuous tie-break: log-prob of the wanted string
            llama_memory_t m = llama_get_memory(cx);
            const uint32_t nb2 = llama_n_batch(cx);

            for (const auto & nc : niah) {
                // prefill everything but the last prompt token
                llama_memory_clear(m, true);
                const size_t npre = nc.prompt.size();
                for (size_t i = 0; i < npre; i += nb2) {
                    const uint32_t n = (uint32_t) std::min<size_t>(nb2, npre - i);
                    llama_batch bb = llama_batch_get_one(const_cast<llama_token *>(nc.prompt.data() + i), n);
                    if (llama_decode(cx, bb) != 0) {
                        return -1e9;
                    }
                }

                // Generate and accept the document if the wanted value appears anywhere in
                // the output - the same lenient rule lod-needle-mk3.sh uses, and the only
                // one that ports across models. (An exact "greedy must emit these tokens"
                // check was tried first: it works on gemma4 and reports a flat,
                // budget-insensitive score on qwen3.6 purely because that model phrases its
                // answer differently. That is a model-specific rule masquerading as a
                // metric.)
                //
                // One forward pass per document, no rewinding. Rewinding and re-feeding
                // tokens is impossible on an M-RoPE model (qwen3.6): the batch validator
                // requires every new batch to start strictly after the position already
                // stored in the cache, and a rewind does not lower that. So positions always
                // come from the cache, and everything is read out of this single pass:
                //   pass/fail  - does the wanted value appear in what greedy generates
                //   tie-break  - log-prob the model puts on STARTING the correct value
                std::vector<llama_token> wt = common_tokenize(cx, " " + nc.want, false, true);

                std::string out;
                bool   ok       = false;
                bool   first    = true;
                double first_lp = -20.0;

                for (int t = 0; t < niah_n; ++t) {
                    const float * lg = llama_get_logits_ith(cx, -1);
                    if (!lg) {
                        break;
                    }
                    llama_token arg = 0;
                    float       mx  = -INFINITY;
                    for (int v = 0; v < n_vocab; ++v) {
                        if (lg[v] > mx) { mx = lg[v]; arg = v; }
                    }
                    if (first && !wt.empty()) {
                        double se = 0.0;
                        for (int v = 0; v < n_vocab; ++v) {
                            se += exp((double)(lg[v] - mx));
                        }
                        first_lp = std::max(-20.0, (double)(lg[wt[0]] - mx) - log(se));
                        first    = false;
                    }
                    if (llama_vocab_is_eog(vocab, arg)) {
                        break;
                    }
                    out += common_token_to_piece(cx, arg);
                    if (out.find(nc.want) != std::string::npos) {
                        ok = true;
                        break;
                    }
                    llama_batch b1 = llama_batch_init(1, 0, 1);
                    common_batch_add(b1, arg, llama_memory_seq_pos_max(m, 0) + 1, { 0 }, true);
                    const int rc1 = llama_decode(cx, b1);
                    llama_batch_free(b1);
                    if (rc1 != 0) {
                        return -1e9;
                    }
                }

                margin += first_lp;
                if (ok) {
                    niah_pass++;
                }
            }

            // the pass count dominates; the log-prob only orders candidates inside a tier and
            // is squashed so it can never buy a pass that was not earned
            const double m2 = niah.empty() ? 0.0 : margin/niah.size();
            return niah_pass + std::max(-0.99, m2/20.0);
        };

        auto score_ctx = [&](llama_context * cx) -> double {
        double total = 0.0;
        int    n_ans = 0;
        guard_nll = 0.0;
        llama_memory_t m2 = llama_get_memory(cx);
        llama_memory_clear(m2, true);
        size_t guard_from = doc.size();
        {
            const uint32_t nb2 = llama_n_batch(cx);
            for (size_t i = 0; i < n_ctxdoc; i += nb2) {
                const uint32_t n    = (uint32_t) std::min<size_t>(nb2, n_ctxdoc - i);
                const bool     last = (i + n >= n_ctxdoc);

                if (!last || n_guard <= 0) {
                    llama_batch bb = llama_batch_get_one(const_cast<llama_token *>(doc.data() + i), n);
                    if (llama_decode(cx, bb) != 0) {
                        return -1e9;
                    }
                    continue;
                }

                // last chunk: ask for logits over its tail, which is the guard segment
                const uint32_t ng = std::min<uint32_t>((uint32_t) n_guard, n);
                guard_from = i + n - ng;

                llama_batch bb = llama_batch_init((int32_t) n, 0, 1);
                for (uint32_t j = 0; j < n; ++j) {
                    common_batch_add(bb, doc[i + j], (llama_pos)(i + j), { 0 }, i + j >= guard_from);
                }
                const int rc = llama_decode(cx, bb);
                llama_batch_free(bb);
                if (rc != 0) {
                    return -1e9;
                }

                if (capture_dense) {
                    dense_lp.assign((size_t) ng*n_vocab, 0.0f);
                }
                int    n_g = 0;
                double acc = 0.0;
                for (size_t t = guard_from; t < doc.size(); ++t) {
                    // logits are addressed by position in the batch, not by output ordinal
                    const float * lg = llama_get_logits_ith(cx, (int32_t)(t - i));
                    if (!lg) {
                        continue;
                    }
                    const size_t g = t - guard_from;
                    float mx = -INFINITY;
                    for (int v = 0; v < n_vocab; ++v) {
                        mx = std::max(mx, lg[v]);
                    }
                    double se = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        se += exp((double)(lg[v] - mx));
                    }
                    const double lse = mx + log(se);

                    if (capture_dense) {
                        float * dst = dense_lp.data() + g*n_vocab;
                        for (int v = 0; v < n_vocab; ++v) {
                            dst[v] = (float)(lg[v] - lse);
                        }
                        n_g++;
                        continue;
                    }
                    if (dense_lp.size() >= (g + 1)*(size_t) n_vocab) {
                        // KL(dense || candidate), which penalises a starved model for being
                        // confidently different as well as for being uncertain
                        const float * ref_lp = dense_lp.data() + g*n_vocab;
                        double kl = 0.0;
                        for (int v = 0; v < n_vocab; ++v) {
                            const double pr = exp((double) ref_lp[v]);
                            if (pr > 1e-9) {
                                kl += pr*((double) ref_lp[v] - ((double) lg[v] - lse));
                            }
                        }
                        acc += kl;
                        n_g++;
                    }
                }
                guard_nll = n_g ? acc/n_g : 0.0;
            }
        }
        for (const auto & pr : pj) {
            const std::string prefix = pr["prefix"].get<std::string>();
            const std::string answer = pr["answer"].get<std::string>();

            // rewind to the end of the document, so every probe sees the same state
            llama_memory_seq_rm(m2, 0, llama_memory_seq_pos_max(m2, 0) + 1, -1);

            std::vector<llama_token> ptok = common_tokenize(cx, prefix, false, true);
            std::vector<llama_token> atok = common_tokenize(cx, answer, false, true);
            if (ptok.empty() || atok.empty()) {
                continue;
            }

            // teacher-force prefix + answer, reading the log-prob of each answer token
            std::vector<llama_token> seq = ptok;
            seq.insert(seq.end(), atok.begin(), atok.end());

            const auto accum = [&](const float * lg, llama_token want) {
                if (!lg) {
                    return;
                }
                float mx = -INFINITY;
                for (int t = 0; t < n_vocab; ++t) {
                    mx = std::max(mx, lg[t]);
                }
                double se = 0.0;
                for (int t = 0; t < n_vocab; ++t) {
                    se += exp((double)(lg[t] - mx));
                }
                total += (double)(lg[want] - mx) - log(se);
                n_ans++;
            };

            if (dec_phase) {
                // one token per llama_decode, so the graph takes the decode branch and the
                // decode budget is what is actually being measured
                for (size_t i = 0; i < seq.size(); ++i) {
                    llama_batch b = llama_batch_init(1, 0, 1);
                    common_batch_add(b, seq[i], llama_memory_seq_pos_max(m2, 0) + 1 + (llama_pos) i, { 0 }, true);
                    const int rc = llama_decode(cx, b);
                    llama_batch_free(b);
                    if (rc != 0) {
                        return -1e9;
                    }
                    if (i + 1 >= ptok.size() && i + 1 < seq.size()) {
                        accum(llama_get_logits_ith(cx, 0), seq[i + 1]);
                    }
                }
            } else {
                llama_batch b = llama_batch_init((int32_t) seq.size(), 0, 1);
                for (size_t i = 0; i < seq.size(); ++i) {
                    common_batch_add(b, seq[i], llama_memory_seq_pos_max(m2, 0) + 1 + (llama_pos) i, { 0 }, true);
                }
                const int rc = llama_decode(cx, b);
                llama_batch_free(b);
                if (rc != 0) {
                    return -1e9;
                }

                // logits at index i predict token i+1
                for (size_t a = 0; a < atok.size(); ++a) {
                    accum(llama_get_logits_ith(cx, (int32_t)(ptok.size() + a - 1)), atok[a]);
                }
            }
        }

        return n_ans ? total/n_ans : -1e9;
        };

        // The profile branch reshapes per ubatch and asserts on a 1-token decode, and the
        // needle objective generates one token at a time. Turn it off now that the layers
        // are known: from here on every context is a plain scoring context.
        unsetenv("LLAMA_LOD_PROFILE");
        llama_context_params cp = common_context_params_to_llama(params);
        cp.cb_eval           = nullptr;
        cp.cb_eval_user_data = nullptr;

        // Everything from here builds its own context per candidate, so the discovery
        // context is dead weight - and it holds a full KV cache, which doubles peak memory
        // for the whole search. `ctx` must not be touched after this point.
        const uint32_t n_ctx_used = llama_n_ctx(ctx);
        llama_init->free_context();
        ctx = nullptr;

        if (search_mode.empty()) {
            if (!niah.empty()) {
                setenv("LLAMA_LOD_TOP_PAGES", eval_top.empty() ? "32" : eval_top.c_str(), 1);
                llama_context * cx = llama_init_from_model(model, cp);
                if (!cx) {
                    LOG_ERR("%s: could not build a scoring context\n", __func__);
                    return 1;
                }
                const double v = score_niah(cx);
                llama_free(cx);
                LOG_INF("\nneedle: %d/%zu documents answered (score %.4f)\n", niah_pass, niah.size(), v);
                llama_backend_free();
                return 0;
            }
            setenv("LLAMA_LOD_TOP_PAGES", eval_top.empty() ? "32" : eval_top.c_str(), 1);
            llama_context * cx = llama_init_from_model(model, cp);
            if (!cx) {
                LOG_ERR("%s: could not build a scoring context\n", __func__);
                return 1;
            }
            const double v = score_ctx(cx);
            llama_free(cx);
            LOG_INF("\nretrieval logprob: %.4f per answer token (%zu probes)\n", v, pj.size());
            LOG_INF("(higher is better; this is the objective a budget search must not regress)\n");
            llama_backend_free();
            return 0;
        }

        // The probe context is rebuilt per candidate, so the one created by common_init is
        // dead weight from here on - and it holds a full KV cache.
        std::vector<int> lod_layers;
        for (const auto & [il, lp] : collector.layers) {
            lod_layers.push_back(il);
            GGML_UNUSED(lp);
        }
        if (lod_layers.empty()) {
            // no profiling pass in probe mode: ask the model which layers LoD is wired into
            for (int il = 0; il < llama_model_n_layer(model); ++il) {
                lod_layers.push_back(il);
            }
        }

        const char * env_search = dec_phase ? "LLAMA_LOD_TOP_PAGES_DECODE" : "LLAMA_LOD_TOP_PAGES";
        int n_eval = 0;

        double last_guard = 0.0;

        auto eval_spec = [&](const std::string & spec) -> double {
            setenv(env_search, spec.c_str(), 1);
            if (dec_phase) {
                // hold prefill at a saturating budget so only the decode budget moves
                setenv("LLAMA_LOD_TOP_PAGES", std::to_string(n_ctx_used).c_str(), 1);
            }
            llama_context * cx = llama_init_from_model(model, cp);
            if (!cx) {
                return -1e9;
            }
            const int64_t t0 = ggml_time_us();
            const double  v  = niah.empty() ? score_ctx(cx) : score_niah(cx);
            llama_free(cx);
            n_eval++;
            last_guard = guard_nll;
            LOG_INF("  [eval %2d] %6.1fs  %s %.4f  %s\n", n_eval, (ggml_time_us() - t0)/1e6,
                    niah.empty() ? "KL" : "niah",
                    niah.empty() ? last_guard : (double) niah_pass,
                    last_guard, spec.size() > 40 ? (spec.substr(0, 37) + "...").c_str() : spec.c_str());
            return v;
        };

        // an allocation is scored by naming the per-layer budgets on top of a default
        auto spec_of = [&](const std::map<int,uint32_t> & a, uint32_t dflt) {
            std::string sp = std::to_string(dflt);
            for (const auto & [il, k] : a) {
                sp += "," + std::to_string(il) + "=" + std::to_string(k);
            }
            return sp;
        };

        const uint32_t base = eval_top.empty() ? 32 : (uint32_t) atoi(eval_top.c_str());
        const std::string base_spec = std::to_string(base);

        // The floor is a real decision, not a formality, and it is NOT free to raise.
        // The objective only measures retrieval, so the search will happily drop a layer
        // retrieval does not need to the bottom of --range - and a layer at 8 pages reads
        // 512 tokens of a 24k document, which long-form generation notices. Measured at a
        // fixed 512 pages on qwen3.6: floor 8 -> coverage 11/12 needle 4/6, floor 24 ->
        // coverage 12/12 needle 2/6. Raising the floor buys coverage with the pages the
        // peaks needed. Neither dominates, so the tool warns instead of choosing.
        if (!range_set && kmin*4 < base) {
            LOG_INF("note: floor is %u pages against a %u baseline. The search will starve\n"
                    "      layers the needle objective does not care about, which can cost\n"
                    "      long-form quality it cannot see. If lod-coverage.sh drops on the\n"
                    "      result, re-run with --range %u:%u and compare both.\n",
                    kmin, base, (base*3)/4, kmax);
        }

        LOG_INF("\n== phase: %s | probes: %zu | baseline budget: %u pages/layer\n",
                dec_phase ? "decode" : "prefill", pj.size(), base);
        LOG_INF("== per-layer sensitivity (starve one layer to %u, keep the rest at %u)\n", kmin, base);

        // capture the dense reference distribution the guard is measured against
        capture_dense = niah.empty();
        if (capture_dense) {
        eval_spec(std::to_string(n_ctx_used));
        capture_dense = false;
        LOG_INF("captured the dense reference over %d guard positions\n", n_guard);
        }

        if (niah.empty() && pj.empty()) {
            LOG_ERR("%s: a search needs --probes or --niah\n", __func__);
            return 1;
        }

        const double ref       = eval_spec(base_spec);
        const double ref_guard = last_guard;
        LOG_INF("baseline retrieval logprob: %.4f   (KL vs dense: %.4f nats over %d positions)\n"
                "\n%-7s %-12s %s\n", ref, ref_guard, n_guard, "layer", "logprob", "cost of starving");

        std::vector<std::pair<double,int>> sens;   // (drop, layer)
        for (int il : lod_layers) {
            const double v = eval_spec(base_spec + "," + std::to_string(il) + "=" + std::to_string(kmin));
            if (v < -1e8) {
                continue;
            }
            sens.push_back({ ref - v, il });
            LOG_INF("%-7d %-12.4f %+.4f\n", il, v, ref - v);
        }
        if (sens.empty()) {
            LOG_ERR("%s: no LoD layers responded\n", __func__);
            return 1;
        }
        std::sort(sens.begin(), sens.end(), [](const auto & a, const auto & b) { return a.first > b.first; });

        LOG_INF("\nranked by cost of starving:\n");
        for (const auto & [d, il] : sens) {
            LOG_INF("  layer %-3d  %+.4f\n", il, d);
        }

        struct cand { std::map<int,uint32_t> alloc; double score; double guard; };
        std::vector<cand> seen;

        std::map<int,uint32_t> best;
        double best_score = ref;
        double best_guard = ref_guard;
        for (const auto & [d, il] : sens) {
            best[il] = base;
            GGML_UNUSED(d);
        }

        if (search_mode == "sens") {
            LOG_INF("\nsensitivity only; spend budget on the top of that list.\n");
        } else if (search_mode == "trim") {
            // Redistribution at a fixed total loses to a flat budget on this model (see the
            // docs), but the same measurement supports the opposite move: start from a
            // budget that is KNOWN to be good, then take pages away from the layers that do
            // not miss them, while retrieval stays within --tie of where it started. The
            // result is cheaper at the same measured quality instead of better at the same
            // cost - which is the question a deployment actually asks.
            LOG_INF("\n== trimming from a uniform %u, keeping retrieval within %.3f of %.4f\n",
                    base, tie, ref);
            const double floor_score = ref - tie;
            for (auto it = sens.rbegin(); it != sens.rend() && n_eval < max_eval; ++it) {
                const int il = it->second;
                uint32_t lo = kmin;
                uint32_t hi = best[il];
                // halve until it hurts, then keep the last budget that did not
                while (lo < hi && n_eval < max_eval) {
                    const uint32_t mid = std::max(kmin, hi/2);
                    if (mid >= hi) {
                        break;
                    }
                    std::map<int,uint32_t> a = best;
                    a[il] = mid;
                    const double v = eval_spec(spec_of(a, base));
                    if (v >= floor_score) {
                        LOG_INF("  layer %-3d %u -> %u pages   retrieval %.4f (kept)\n", il, hi, mid, v);
                        best[il]   = mid;
                        best_score = v;
                        best_guard = last_guard;
                        hi         = mid;
                    } else {
                        LOG_INF("  layer %-3d %u -> %u pages   retrieval %.4f (too far, stop)\n", il, hi, mid, v);
                        break;
                    }
                }
            }
        } else {
            // ---- budget-constrained shaping.
            // Keep the TOTAL pages equal to the uniform baseline and redistribute them by
            // measured importance. alpha=0 is the uniform baseline itself, so a shaped
            // allocation only wins if it actually beats uniform at the same cost - which is
            // the only claim worth making.
            const uint32_t n_l    = (uint32_t) sens.size();
            const uint32_t budget = base * n_l;
            const uint32_t step   = 8;

            auto shape = [&](double alpha) {
                std::vector<double> w(n_l);
                double sw = 0.0;
                for (uint32_t i = 0; i < n_l; ++i) {
                    w[i] = pow(std::max(sens[i].first, 0.0) + 1e-3, alpha);
                    sw  += w[i];
                }
                std::map<int,uint32_t> a;
                uint32_t spent = 0;
                for (uint32_t i = 0; i < n_l; ++i) {
                    uint32_t k = (uint32_t) llround(budget*w[i]/sw/step)*step;
                    k = std::max(kmin, std::min(kmax, k));
                    a[sens[i].second] = k;
                    spent += k;
                }
                // give back / take back the rounding drift at the important end
                for (uint32_t i = 0; spent > budget && i < n_l; ++i) {
                    const int il = sens[n_l-1-i].second;
                    while (spent > budget && a[il] > kmin) { a[il] -= step; spent -= step; }
                }
                for (uint32_t i = 0; spent + step <= budget && i < n_l; ++i) {
                    const int il = sens[i].second;
                    while (spent + step <= budget && a[il] + step <= kmax) { a[il] += step; spent += step; }
                }
                return a;
            };

            LOG_INF("\n== shaping at a fixed total of %u pages (%u layers x %u)\n", budget, n_l, base);
            for (double alpha : { 0.5, 1.0, 2.0, 4.0 }) {
                std::map<int,uint32_t> a = shape(alpha);
                const double v  = eval_spec(spec_of(a, base));
                const double dg = ref_guard - last_guard;   // positive = less divergent
                const bool   ok = !niah.empty() || dg > -guard_tol;
                LOG_INF("  alpha %-4.1f  logprob %-10.4f KL %.4f (%+.4f) %s\n", alpha, v, last_guard, dg,
                        !ok ? "(REJECTED: guard)" : "(kept)");
                if (ok) {
                    seen.push_back({ a, v, last_guard });
                }
            }

            // refinement continues from the best shape that survived the guard, not from
            // the uniform start it began at
            for (const auto & c : seen) {
                if (c.score > best_score) {
                    best       = c.alloc;
                    best_score = c.score;
                    best_guard = c.guard;
                }
            }

            // ---- local refinement: move one step from the cheapest layer to the dearest.
            // Every move keeps the total constant, so an accepted move is a strict win at
            // equal cost.
            LOG_INF("\n== refinement (constant total, %u pages per move)\n", step);
            // only the extremes of the ranking are worth trying, so an unproductive round
            // costs a bounded number of evaluations instead of n_layers^2
            const uint32_t nend = std::min<uint32_t>(4, n_l);
            for (int round = 0; round < 6 && n_eval < max_eval; ++round) {
                bool moved = false;
                for (uint32_t g = 0; g < nend && !moved; ++g) {
                    for (uint32_t t = n_l; t-- > n_l - nend && !moved; ) {
                        const int gi = sens[g].second;      // give to (important)
                        const int ti = sens[t].second;      // take from (cheap)
                        if (gi == ti || best[gi] + step > kmax || best[ti] < kmin + step) {
                            continue;
                        }
                        std::map<int,uint32_t> a = best;
                        a[gi] += step;
                        a[ti] -= step;
                        if (n_eval >= max_eval) {
                            LOG_INF("  evaluation budget reached (--max-eval %d)\n", max_eval);
                            break;
                        }
                        const double v = eval_spec(spec_of(a, base));
                        if (niah.empty() && ref_guard - last_guard <= -guard_tol) {
                            continue;
                        }
                        seen.push_back({ a, v, last_guard });
                        if (v > best_score + 1e-4) {
                            LOG_INF("  layer %d -> %d : %.4f -> %.4f (accepted)\n", ti, gi, best_score, v);
                            best_score = v;
                            best       = a;
                            best_guard = last_guard;
                            moved      = true;
                        }
                    }
                }
                if (!moved) {
                    LOG_INF("  no further improvement\n");
                    break;
                }
            }
        }

        if (!seen.empty()) {
            double top = best_score;
            for (const auto & c : seen) {
                top = std::max(top, c.score);
            }
            // among everything within --tie of the best retrieval, take the best guard:
            // 0.004 nats of retrieval is not worth 1.0 nats of language modelling
            for (const auto & c : seen) {
                if (c.score >= top - tie && c.guard < best_guard) {
                    best       = c.alloc;
                    best_score = c.score;
                    best_guard = c.guard;
                }
            }
            if (best_score < top - tie) {
                for (const auto & c : seen) {
                    if (c.score > best_score) {
                        best = c.alloc; best_score = c.score; best_guard = c.guard;
                    }
                }
            }
            LOG_INF("\nselected within %.3f nats of the best retrieval (%.4f), on the guard\n", tie, top);
        }

        uint32_t total_pages = 0;
        for (const auto & [il, k] : best) {
            total_pages += k;
        }

        LOG_INF("\n== result (%d evaluations)\n", n_eval);
        LOG_INF("uniform %u/layer : retrieval %.4f   KL-vs-dense %.4f\n", base, ref, ref_guard);
        LOG_INF("searched         : retrieval %.4f   KL-vs-dense %.4f  (%+.4f retrieval, %+.4f KL)\n"
                "                   at the same total of %u pages\n",
                best_score, best_guard, best_score - ref, best_guard - ref_guard, total_pages);
        if (best_score <= ref + 1e-4) {
            LOG_INF("the search did NOT beat uniform - keep --lod-top-pages %u and spend the\n"
                    "budget uniformly. This is a real outcome, not a failure of the tool.\n", base);
        }
        for (const auto & [il, k] : best) {
            LOG_INF("  layer %-3d %u pages\n", il, k);
        }

        // ---- emit / update the config, preserving the other phase if it is already there
        nlohmann::json cfg;
        {
            std::ifstream ex(out);
            if (ex) {
                try { cfg = nlohmann::json::parse(ex, nullptr, true, true); } catch (...) { cfg = nlohmann::json(); }
            }
        }
        const char * page_env0 = getenv("LLAMA_LOD_PAGE_SIZE");
        cfg["page_size"]           = page_env0 ? atoi(page_env0) : 64;
        cfg["n_ctx_calibrated"]    = n_ctx_used;
        cfg["n_tokens_calibrated"] = doc.size();
        cfg["model"]               = params.model.path;
        cfg["objective"]           = "retrieval_logprob";
        cfg[dec_phase ? "decode_logprob" : "prefill_logprob"]   = best_score;
        cfg[dec_phase ? "decode_uniform" : "prefill_uniform"]   = ref;
        cfg["guard_logprob"]         = best_guard;
        cfg["guard_logprob_uniform"] = ref_guard;
        cfg["budget_total"]        = total_pages;
        cfg["default"][dec_phase ? "decode" : "prefill"] = base;
        for (const auto & [il, k] : best) {
            cfg["layers"][std::to_string(il)][dec_phase ? "decode" : "prefill"] = k;
        }
        {
            std::ofstream of(out);
            if (!of) {
                LOG_ERR("%s: cannot write %s\n", __func__, out.c_str());
                return 1;
            }
            of << cfg.dump(2) << "\n";
        }
        LOG_INF("\nwrote %s\nuse it with:  --lod --lod-config %s\n", out.c_str(), out.c_str());

        llama_backend_free();
        return 0;
    }

    const std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    if (tokens.size() < 128) {
        LOG_ERR("%s: calibration prompt is too short (%zu tokens) - pass a long -f file\n",
                __func__, tokens.size());
        return 1;
    }

    const uint32_t n_batch = llama_n_batch(ctx);
    LOG_INF("%s: profiling over %zu tokens\n", __func__, tokens.size());

    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const uint32_t n = (uint32_t) std::min<size_t>(n_batch, tokens.size() - i);

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + i), n);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed at offset %zu\n", __func__, i);
            return 1;
        }
        LOG_INF("  processed %zu / %zu\n", std::min(i + n, tokens.size()), tokens.size());
    }

    if (collector.layers.empty()) {
        LOG_ERR("%s: no profile tensors captured - is this a model with LoD layers?\n", __func__);
        return 1;
    }

    // ---- report the curves and pick the budgets
    LOG_INF("\n%-6s %-7s %-35s %-8s %s\n", "layer", "pages",
            "capture at k = 8/16/32/64/128/256", "chosen", "capture  lift(x random)");

    std::map<int, uint32_t>           chosen;
    std::map<int, std::vector<float>> curves;   // concave envelope value at every k
    uint32_t total = 0;

    for (const auto & [il, lp] : collector.layers) {
        if (lp.mass.empty() || lp.score.empty() || lp.mass.size() != lp.score.size()) {
            continue;
        }
        const std::vector<float> cum = capture_curve(lp);
        const uint32_t n_pages = (uint32_t) lp.mass.size();
        const uint32_t hi      = std::min(kmax, n_pages);

        // Buy pages while the next ones are worth clearly more than a random page. Stopping
        // at a fixed capture target instead drives diffuse layers to ~85% of all pages,
        // which is both useless as a config and pointless: mass the selector cannot
        // localise is mass the page means already represent.
        //
        // The marginal gains are NOT monotone - a layer can sit on a plateau and then jump
        // (layer 59 of gemma4 goes 0.155 at k=32 to 0.585 at k=64), so a local slope test
        // stops far too early. Walk the CONCAVE ENVELOPE of the curve instead: its segment
        // slopes are decreasing by construction and a segment's slope is the true value of
        // the pages it spans.
        std::vector<uint32_t> ks;
        std::vector<float>    cs;
        for (uint32_t c = 0; c <= hi; c += round_to) {
            // keep the upper hull: drop points that the previous two make non-concave
            while (ks.size() >= 2) {
                const size_t n2 = ks.size();
                const float s_prev = (cs[n2-1] - cs[n2-2])/(float)(ks[n2-1] - ks[n2-2]);
                const float s_new  = (cum[c]   - cs[n2-2])/(float)(c        - ks[n2-2]);
                if (s_new < s_prev) {
                    break;
                }
                ks.pop_back();
                cs.pop_back();
            }
            ks.push_back(c);
            cs.push_back(cum[c]);
        }

        uint32_t k = std::min(kmin, hi);
        for (size_t i = 1; i < ks.size(); ++i) {
            if (cs[i-1] >= target) {
                break;
            }
            const float slope = (cs[i] - cs[i-1])/(float)(ks[i] - ks[i-1]);
            if (slope*n_pages < min_lift) {
                break;
            }
            k = ks[i];
        }
        k = std::max(k, std::min(kmin, hi));

        const float lift = k > 0 ? cum[k]/((float) k/n_pages) : 0.0f;

        chosen[il] = k;
        total += k;
        curves[il] = cum;

        auto at = [&](uint32_t q) { return q <= n_pages ? cum[q] : cum[n_pages]; };
        LOG_INF("%-6d %-7u %.3f %.3f %.3f %.3f %.3f %.3f   %-8u %.3f    %.1fx\n",
                il, n_pages, at(8), at(16), at(32), at(64), at(128), at(256), k, cum[k], lift);
    }

    // ---- the two sides of the trade-off, in units a user can reason about.
    // cost: pages are linear in time for both prefill (leaf rows read) and decode (leaf
    // walk), so "share of the context read exactly" is a faithful cost axis.
    // quality: the capture of the worst layer, because a starved layer corrupts what every
    // later layer reads.
    const uint32_t n_layers = (uint32_t) curves.size();
    const uint32_t n_pages0 = curves.empty() ? 1 : (uint32_t)(curves.begin()->second.size() - 1);

    const auto worst_capture = [&](const std::map<int, uint32_t> & alloc) {
        float w = 1.0f;
        for (const auto & [il, k] : alloc) {
            w = std::min(w, curves.at(il)[k]);
        }
        return w;
    };
    const auto mean_capture = [&](const std::map<int, uint32_t> & alloc) {
        float m = 0.0f;
        for (const auto & [il, k] : alloc) {
            m += curves.at(il)[k];
        }
        return alloc.empty() ? 0.0f : m/alloc.size();
    };
    // spend a total budget so that the worst layer is as good as possible
    const auto allocate_cost = [&](uint32_t budget) {
        std::map<int, uint32_t> alloc;
        for (const auto & [il, k] : chosen) {
            alloc[il] = std::min(kmin, n_pages0);
            GGML_UNUSED(k);
        }
        uint32_t spent = std::min(kmin, n_pages0)*n_layers;
        while (spent + round_to <= budget) {
            int   worst = -1;
            float worst_cap = 1e9f;
            for (const auto & [il, k] : alloc) {
                if (k + round_to > n_pages0) {
                    continue;
                }
                if (curves.at(il)[k] < worst_cap) {
                    worst_cap = curves.at(il)[k];
                    worst     = il;
                }
            }
            if (worst < 0) {
                break;
            }
            alloc[worst] += round_to;
            spent        += round_to;
        }
        return alloc;
    };
    // cheapest allocation that brings every layer to a capture floor
    const auto allocate_quality = [&](float floor_v) {
        std::map<int, uint32_t> alloc;
        for (const auto & [il, cum] : curves) {
            uint32_t k = std::min(kmin, n_pages0);
            for (uint32_t c = k; c <= std::min(kmax, n_pages0); c += round_to) {
                k = c;
                if (cum[c] >= floor_v) {
                    break;
                }
            }
            alloc[il] = k;
        }
        return alloc;
    };
    const auto report = [&](const std::map<int, uint32_t> & alloc) {
        uint32_t sum = 0;
        for (const auto & [il, k] : alloc) { sum += k; GGML_UNUSED(il); }
        LOG_INF("%-7u %-9.1f %-11.2f %-9.3f %.3f\n", sum, (double) sum/n_layers,
                100.0*sum/(double)(n_layers*n_pages0), worst_capture(alloc), mean_capture(alloc));
        return sum;
    };

    // ---- how big is the UNION of the per-block selections? If per-32-query-block selection
    // can be served by gathering that union (union x ps leaf rows) instead of reading the
    // whole span, mask-grade selection becomes affordable at gather-like cost.
    {
        bool have = false;
        for (const auto & [il, lp] : collector.layers) {
            if (!lp.qscore.empty()) { have = true; break; }
            GGML_UNUSED(il);
        }
        if (have) {
            LOG_INF("\nunion of per-32-query-block selections (pages actually needed):\n");
            LOG_INF("%-6s %-8s %s\n", "layer", "pages", "k=8 / 16 / 32 / 64  (union, x of k)");
            for (const auto & [il, lp] : collector.layers) {
                if (lp.qscore.empty()) {
                    continue;
                }
                const int64_t n_tok   = lp.n_tok;
                const int64_t n_pages = (int64_t) lp.qscore.size()/std::max<int64_t>(1, n_tok);
                const int64_t n_blk   = (n_tok + 31)/32;

                std::string row;
                for (uint32_t k : {8u, 16u, 32u, 64u}) {
                    std::vector<char> seen(n_pages, 0);
                    for (int64_t b = 0; b < n_blk; ++b) {
                        // pool this block's queries, then take its top-k
                        std::vector<float> pooled(n_pages, -INFINITY);
                        for (int64_t q = b*32; q < std::min<int64_t>((b+1)*32, n_tok); ++q) {
                            for (int64_t p = 0; p < n_pages; ++p) {
                                pooled[p] = std::max(pooled[p], lp.qscore[q + p*n_tok]);
                            }
                        }
                        std::vector<int64_t> idx(n_pages);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::partial_sort(idx.begin(), idx.begin() + std::min<int64_t>(k, n_pages), idx.end(),
                                [&](int64_t a, int64_t c) { return pooled[a] > pooled[c]; });
                        for (uint32_t i = 0; i < std::min<int64_t>(k, n_pages); ++i) {
                            seen[idx[i]] = 1;
                        }
                    }
                    const int64_t u = std::count(seen.begin(), seen.end(), (char) 1);
                    char tmp[64];
                    snprintf(tmp, sizeof(tmp), " %4lld(%.1fx)", (long long) u, (double) u/k);
                    row += tmp;
                }
                LOG_INF("%-6d %-8lld%s\n", il, (long long) n_pages, row.c_str());
            }
            LOG_INF("\n(a union close to the page count means per-block selection cannot be\n"
                    " served by a compact gather - that is the regime mask exists for)\n");
        }
    }

    if (mode == "pareto") {
        LOG_INF("\n%-7s %-9s %-11s %-9s %s\n", "pages", "per layer", "ctx read %", "worst cap", "mean cap");
        // widen the SWEEP step as the curve flattens, but never the allocation granularity:
        // coarsening round_to makes the greedy lumpy and the frontier stops being monotone
        uint32_t step = round_to;
        for (uint32_t b = kmin*n_layers; b <= std::min<uint32_t>(kmax, n_pages0)*n_layers; b += step*n_layers) {
            report(allocate_cost(b));
            if (b >= 64*n_layers && (b/n_layers) % 64 == 0) {
                step *= 2;
            }
        }
        LOG_INF("\npick a row and re-run with --target cost=<pages> or --target quality=<worst cap>\n");
        llama_backend_free();
        return 0;
    }

    if (mode == "cost" || mode == "quality") {
        std::map<int, uint32_t> alloc;
        if (mode == "quality") {
            alloc = allocate_quality(quality_floor);
        } else {
            uint32_t budget;
            if (!cost_arg.empty() && cost_arg.back() == '%') {
                budget = (uint32_t)(atof(cost_arg.c_str())/100.0*n_layers*n_pages0);
            } else {
                budget = (uint32_t) atoi(cost_arg.c_str());
            }
            alloc = allocate_cost(budget);
        }
        LOG_INF("\n%-7s %-9s %-11s %-9s %s\n", "pages", "per layer", "ctx read %", "worst cap", "mean cap");
        total = report(alloc);
        chosen = alloc;
        LOG_INF("\n");
        for (const auto & [il, k] : chosen) {
            LOG_INF("  layer %-3d %-5u pages  capture %.3f\n", il, k, curves.at(il)[k]);
        }
    }

    // ---- fixed total budget: equalise capture instead of thresholding each layer alone.
    // With one page set shared across the whole ubatch the capture is low everywhere, so a
    // lift threshold starves every layer; what actually correlates with end quality is how
    // badly the WORST layer is served (measured: uniform 16 pages -> 5/12 coverage,
    // uniform 32 -> 11/12, and the capture roughly doubles between them).
    if (budget_total > 0 && !curves.empty()) {
        for (auto & [il, k] : chosen) {
            k = kmin;
        }
        uint32_t spent = kmin*(uint32_t) chosen.size();

        while (spent + round_to <= (uint32_t) budget_total) {
            int    worst = -1;
            double worst_cap = 1e9;
            for (const auto & [il, k] : chosen) {
                const auto & cum = curves[il];
                if (k + round_to >= cum.size()) {
                    continue;
                }
                if (cum[k] < worst_cap) {
                    worst_cap = cum[k];
                    worst     = il;
                }
            }
            if (worst < 0) {
                break;
            }
            chosen[worst] += round_to;
            spent         += round_to;
        }
        total = spent;

        LOG_INF("\nequalised allocation for a total of %u pages:\n", total);
        for (const auto & [il, k] : chosen) {
            LOG_INF("  layer %-3d %-5u pages  capture %.3f\n", il, k, curves[il][k]);
        }
    }

    // ---- emit the config
    FILE * f = fopen(out.c_str(), "w");
    if (!f) {
        LOG_ERR("%s: cannot write %s\n", __func__, out.c_str());
        return 1;
    }
    const char * page_env = getenv("LLAMA_LOD_PAGE_SIZE");
    fprintf(f, "{\n");
    fprintf(f, "  \"page_size\": %d,\n", page_env ? atoi(page_env) : 64);
    fprintf(f, "  \"n_ctx_calibrated\": %u,\n", llama_n_ctx(ctx));
    fprintf(f, "  \"n_tokens_calibrated\": %zu,\n", tokens.size());
    fprintf(f, "  \"target_capture\": %.4f,\n", target);
    fprintf(f, "  \"min_lift\": %.3f,\n", min_lift);
    fprintf(f, "  \"budget_total\": %u,\n", total);
    if (!curves.empty()) {
        float worst = 1.0f, mean = 0.0f;
        for (const auto & [il, k] : chosen) {
            worst = std::min(worst, curves.at(il)[k]);
            mean += curves.at(il)[k];
        }
        fprintf(f, "  \"ctx_read_exactly_pct\": %.2f,\n",
                100.0*total/(double)(curves.size()*(curves.begin()->second.size() - 1)));
        fprintf(f, "  \"capture_worst\": %.4f,\n", worst);
        fprintf(f, "  \"capture_mean\": %.4f,\n", chosen.empty() ? 0.0f : mean/chosen.size());
    }
    fprintf(f, "  \"model\": \"%s\",\n", params.model.path.c_str());
    fprintf(f, "  \"layers\": {\n");
    bool first = true;
    for (const auto & [il, k] : chosen) {
        fprintf(f, "%s    \"%d\": {\"prefill\": %u}", first ? "" : ",\n", il, k);
        first = false;
    }
    fprintf(f, "\n  }\n}\n");
    fclose(f);

    LOG_INF("\nwrote %s (total budget %u pages over %zu layers)\n", out.c_str(), total, chosen.size());
    LOG_INF("use it with:  --lod --lod-config %s\n", out.c_str());

    llama_backend_free();
    return 0;
}
