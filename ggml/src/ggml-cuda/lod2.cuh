#include "common.cuh"

void ggml_cuda_lod2_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_lod2_attn  (ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_lod2_supported(const ggml_tensor * dst);
