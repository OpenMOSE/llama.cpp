from gguf import GGUFReader

reader = GGUFReader("/home/client/Projects/llm/RWKV-GLM-4.7-Flash-exp/RWKV-GLM-4.7-Flash-Exp-64x3.0B-BF16.gguf")

for tensor in reader.tensors:
    print(f"{tensor.name}: shape={tensor.shape}, type={tensor.tensor_type.name}")