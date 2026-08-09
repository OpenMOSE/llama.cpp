#include "common.cuh"

void ggml_cuda_lod_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_lod_attn_supported(const ggml_tensor * dst);
