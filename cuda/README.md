> **Disclaimer**
> This README was drafted with help from AI and should be updated as the project evolves.
> The CUDA notes were translated into text with AI from my handwritten notes. The original handwritten notes are in `notes/handwritten/`.

# CUDA Practice

CUDA learning and matrix-multiplication kernel practice.

## Status

Multiple matmul kernels implemented in `mat_mult/code/` and benchmarked against cuBLAS:

- `mat_mul_base` — naive, one thread per output element.
- `mat_mul_tiled` — shared-memory tiling.
- `mat_mul_regtile` — register tiling.
- `mat_mul_float4` — float4 vectorized loads.
- `mat_mul_cublas` — cuBLAS reference.

Naive and tiled kernels profiled with Nsight Compute (reports in `mat_mult/testing/ncu-rep/`), including roofline analysis. Testing write-ups are in `notes/testing_notes/` (Phase 1 naive, Phase 2 tiled).

Handwritten study notes in `notes/handwritten/`: CUDA basics, GPU architecture, roofline analysis, benchmarking best practices, and naive/tiled matmul.

## Next steps

- Continue profiling the vectorized and register-tiled kernels.
- Extend benchmark notes to the remaining kernels.
