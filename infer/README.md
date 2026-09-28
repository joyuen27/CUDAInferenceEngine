> **Disclaimer**
> This README was drafted with help from AI and should be updated as the project evolves.

# GPT-2 Inference Engine

A from-scratch GPT-2 (medium) inference engine in C++, built with Bazel.

## Status

Working end-to-end greedy text generation. Implemented components:

- `embedding/` — token + position embedding and unembedding.
- `transformer/layer_norm/` — LayerNorm.
- `transformer/attention/` — multi-head attention, with a KV-cached path.
- `transformer/mlp/` — MLP; runs on a CUDA kernel (`cuda/llm_mat_mul`, float4 vectorized).

KV caching is working (prefill + single-token decode). The MLP runs on the GPU; the rest of the pipeline runs on CPU. Weights are exported from HuggingFace via `model/model.py`, and outputs are validated against PyTorch (`model/testing/`). TTFT and tokens/s are reported per run.

## Build & run

```bash
bash run.sh   # generates tokens, then runs bazel-bin/infer_bin
```

## Next steps

Move more of the pipeline (attention, LayerNorm) onto the GPU.
