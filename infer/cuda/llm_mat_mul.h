#pragma once

#ifndef LLM_MAT_MUL_H
#define LLM_MAT_MUL_H

// Computes C = A @ B where
//   A is [M x K] (row-major),
//   B is [K x N] (row-major, i.e. the NON-transposed weight layout),
//   C is [M x N] (row-major).
// M, N, K may be any positive size (the kernel is bounds-checked).
void mat_mul_float4(const float* a, const float* b, float* c, int M, int N, int K);

#endif
