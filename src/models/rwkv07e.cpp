#include "models.h"


//hxa07d inference code
llm_build_rwkv07e::llm_build_rwkv07e(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {        


        ggml_tensor * cur;
        ggml_tensor * inpL;
        ggml_tensor * v_first = nullptr;
        // ggml_tensor * k_first = nullptr;

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;
        const auto n_head_kv = hparams.n_head_kv_;
        const auto n_head_att = hparams.n_head_att_;


        inpL = build_inp_embd(model.tok_embd);

        ggml_tensor * inp_pos = nullptr;

        if (hparams.nope_in_rwkv == false){
            inp_pos = build_inp_pos();
        }
        auto * inp_hybrid = build_inp_mem_hybrid();
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        

        for (int il = 0; il < n_layer; ++il) {

            ggml_tensor * inpSA = inpL;

            // norm
            ggml_tensor * att_norm = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);

            bool IsRWKV = hparams.is_rwkv(il);

            
            if (IsRWKV)
            {
                // Implementation RWKV07e
                cur = att_norm;
             

                llm_graph_input_rs * inp = inp_hybrid->get_recr();
                const auto * mctx_cur = inp->mctx;

                const auto n_tokens = ubatch.n_tokens;
                const auto n_seqs = ubatch.n_seqs;
                const auto n_embd = hparams.n_embd;
                const auto head_size = hparams.wkv_head_size;
                const auto n_seq_tokens = ubatch.n_seq_tokens;

                const auto kv_head = mctx_cur->get_head();
                const auto & layer = model.layers[il];

                ggml_tensor * x = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

                ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, x);
                ggml_tensor * k = build_lora_mm(layer.time_mix_key, x);
                ggml_tensor * v = build_lora_mm(layer.time_mix_value, x);

                if (layer.time_mix_receptance_b) { r = ggml_add(ctx0, r, layer.time_mix_receptance_b); }
                if (layer.time_mix_key_b) { k = ggml_add(ctx0, k, layer.time_mix_key_b); }
                if (layer.time_mix_value_b) { v = ggml_add(ctx0, v, layer.time_mix_value_b); }


                //reshape -> [B*T,H,N] 
                r = ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);
                k = ggml_reshape_3d(ctx0, k, head_size, n_head_kv, n_tokens);

                if (hparams.enable_qk_norm == true){
                    //receptance RMS norm
                    r = build_norm(r, model.layers[il].attn_r_norm, NULL, LLM_NORM_RMS, il);

                    //key RMS norm
                    k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                }

                if (hparams.nope_in_rwkv == false){
                    r = ggml_rope_ext(
                            ctx0, r, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                
                    k = ggml_rope_ext(
                            ctx0, k, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                }

                // calc decay with softplus style

                r = ggml_reshape_2d(ctx0, r, head_size*n_head, n_tokens);

                // step1 matmul with tanh
                ggml_tensor * w = ggml_add(
                    ctx0,
                    ggml_mul_mat(ctx0, layer.time_mix_w2, ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer.time_mix_w1, r))),
                    layer.time_mix_w0
                );

                struct ggml_tensor * w32 = ggml_cast(ctx0, w, GGML_TYPE_F32);

                // step2 logsigmoid(w_pre) - 0.5 ->  log( sigmoid(w)*exp(-0.5) )
                const float C = 0.6065306597126334f; // exp(-0.5)
                struct ggml_tensor * s      = ggml_sigmoid_inplace(ctx0, w32);             // σ(w)
                struct ggml_tensor * s_c    = ggml_scale_inplace(ctx0, s, C);              // σ(w) * e^{-1/2}
                struct ggml_tensor * log_neglog_w  = ggml_log_inplace(ctx0, s_c);                 // log(σ(w)*e^{-1/2})

                //already fp32 tensor
                w = ggml_exp(ctx0, ggml_neg_inplace(ctx0, ggml_exp(ctx0, log_neglog_w)));








                if (n_head_kv != 0 && n_head_kv != n_head) {
                    GGML_ASSERT(n_head % n_head_kv == 0);
                    v = ggml_reshape_3d(ctx0, v, head_size, n_head_kv, n_tokens);
                    k = ggml_reshape_3d(ctx0, k, head_size, n_head_kv, n_tokens);
        
                }

                
                // residual connection before expand tensor
                if (il==0) {
                    v_first = v;
                } else {
                    // Add the first layer value as a residual connection.
                    v = ggml_add(ctx0, v,
                        ggml_mul(ctx0,
                            ggml_sub(ctx0, v_first, v),
                            ggml_reshape_3d(ctx0, 
                                ggml_sigmoid(ctx0, ggml_add(ctx0,
                                        ggml_mul_mat(ctx0, layer.time_mix_v2, ggml_mul_mat(ctx0, layer.time_mix_v1, x)),
                                        layer.time_mix_v0
                                    )
                                ),
                                head_size, n_head_kv, n_tokens
                            )
                        )
                    );
                }
               
                if (n_head_kv != 0 && n_head_kv != n_head) {
                    ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
                    v = ggml_reshape_4d(ctx0, v, head_size,1, n_head_kv, n_tokens);
                    k = ggml_reshape_4d(ctx0, k, head_size,1, n_head_kv, n_tokens);
                    k = ggml_repeat(ctx0, k, tmp);
                    v = ggml_repeat(ctx0, v, tmp);
                    
                }

                 
                //receptance based gate in 07e
                ggml_tensor * g = ggml_mul_mat(ctx0, layer.time_mix_g2, ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.time_mix_g1, r)));

                //receptance based iclr in 07e
                ggml_tensor * a = ggml_sigmoid(ctx0,
                    ggml_add(
                        ctx0,
                        ggml_mul_mat(ctx0, layer.time_mix_a2, ggml_mul_mat(ctx0, layer.time_mix_a1, r)),
                        layer.time_mix_a0
                    )
                );


                r = ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);
                w = ggml_reshape_3d(ctx0, w, head_size, n_head, n_tokens);
                k = ggml_reshape_3d(ctx0, k, head_size, n_head, n_tokens);

                //Added VLoRA in 07e
                v = ggml_add(
                        ctx0,
                        ggml_reshape_3d(ctx0, v, head_size, n_head, n_tokens),
                        ggml_scale(ctx0,
                                           ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.time_mix_dv2, ggml_mul_mat(ctx0, layer.time_mix_dv1, x)),
                                                           head_size, n_head, n_tokens)
                                           ,0.5
                                  )
                    );
                
                a = ggml_reshape_3d(ctx0, a, head_size, n_head, n_tokens);
                ggml_tensor * kk = ggml_l2_norm(ctx0, k, 1e-12);

                //old k = k * (1 + (a-1) * self.k_a)
                //new k = k * (1.0 - w + a)
                //->  k + k * (a-w)

                k = ggml_add(ctx0, k, ggml_mul(ctx0, k, ggml_sub(ctx0, a, w ) ));
                

                ggml_tensor * wkv_state = build_rs(
                        inp, mctx_cur->get_s_l(il),
                        hparams.n_embd_s(), n_seqs);

                // //currently need fp32
                // w32   = ggml_cast(ctx0, w, GGML_TYPE_F32);
                // ggml_tensor * w_exp = ggml_exp(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, w32)));
                // w     = ggml_cast(ctx0, w_exp, w->type);
               

                ggml_tensor * wkv_output = ggml_rwkv_wkv7(ctx0, r, w, k, v, ggml_neg(ctx0, kk), ggml_mul(ctx0, kk, a), wkv_state);
                        
                cur = ggml_view_1d(ctx0, wkv_output, (head_size*n_head) * n_tokens, 0);
                wkv_state = ggml_view_1d(ctx0, wkv_output, (head_size*n_head) * head_size * n_seqs, (head_size*n_head) * n_tokens * sizeof(float));

                ggml_build_forward_expand(
                        gf,
                        ggml_cpy(
                            ctx0,
                            wkv_state,
                            ggml_view_1d(
                                ctx0,
                                mctx_cur->get_s_l(il),
                                hparams.n_embd_s() * n_seqs,
                                hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il))
                                )
                            )
                        );

            
                cur = ggml_reshape_2d(ctx0, cur, (head_size*n_head), n_tokens);

                cur = ggml_scale(ctx0, cur, 1.0f / sqrtf(float(head_size)));

                cur = ggml_mul(ctx0, cur, g);

                cur = build_lora_mm(layer.time_mix_output, cur);

            }
            else
            {
                //079 uses NoPE Attention
                const auto & layer = model.layers[il];

                
                // self_attention
                const auto head_size = hparams.wkv_head_size;
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, att_norm);
                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, att_norm);
                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, att_norm);

                Qcur = ggml_reshape_3d(ctx0, Qcur, head_size, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, head_size, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, head_size, n_head_kv, n_tokens);

                if (layer.wq_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
                }
                if (layer.wk_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
                }
                if (layer.wv_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
                }

                if (hparams.enable_qk_norm == true){
                    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                }


                cur = build_attn(inp_hybrid->get_attn(),
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(head_size)), il);

              
            }
            //below same qwen3 ffn code
          
 
            if (il == n_layer - 1 && inp_out_ids) {
                //exit(1);
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;

            // if (il == 16){
            //     exit(1);
            // }

            
        }
        

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }