#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <math.h>
#include "mlp.h"
#include "llm_mat_mul.h"

void cpu_mlp(float* ln_embd, float* fc_w, float* fc_b, float* proj_w, float* proj_b,
            int32_t n_embd, int32_t n_tokens, float* mlp_delta)
{
    // FC Matrix Multiply
    float* fc_out = (float*) malloc (4 * n_embd * n_tokens * sizeof(float));
    for (int t = 0; t < n_tokens; t++) {
        //Get row in weight matrix
        for (int r = 0; r < 4 * n_embd; r++) {
            float accum = fc_b[r];
            //Get Token Embedding and dot product with row
            for (int i = 0; i < n_embd; i++) {
                accum += ln_embd[t * n_embd + i] * fc_w[r * n_embd + i];
            }
            fc_out[t * 4 * n_embd + r] = accum;
        }
    }

    //GeLu
    for (int i = 0; i < 4 * n_embd * n_tokens; i++) {
        fc_out[i] = gelu(fc_out[i]);
    }

    // Proj Matrix Multiply
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < n_embd; r++) {
            float accum = proj_b[r];
            for (int i = 0; i < 4 * n_embd; i++) {
                accum += fc_out[t * 4 * n_embd + i] * proj_w[r * 4 * n_embd + i];
            }
            mlp_delta[t * n_embd + r] = accum;
        }
    }

    free(fc_out);
}

// GPU MLP: same math as cpu_mlp but the two matmuls run on the GPU via mat_mul_float4.
// mat_mul_float4 computes C = A @ B with B in NON-transposed [K x N] layout, so fc_w and
// proj_w here are the UN-transposed weights (mlp_fc_w_ut / mlp_proj_w_ut) fed in directly.
void gpu_mlp(float* ln_embd, float* fc_w, float* fc_b, float* proj_w, float* proj_b,
            int32_t n_embd, int32_t n_tokens, float* mlp_delta)
{
    int32_t hidden = 4 * n_embd;

    // FC: fc_out[n_tokens x hidden] = ln_embd[n_tokens x n_embd] @ fc_w[n_embd x hidden]
    float* fc_out = (float*) malloc((size_t) hidden * n_tokens * sizeof(float));
    mat_mul_float4(ln_embd, fc_w, fc_out, n_tokens, hidden, n_embd);

    // Add bias + GeLU
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < hidden; r++) {
            fc_out[t * hidden + r] = gelu(fc_out[t * hidden + r] + fc_b[r]);
        }
    }

    // Proj: mlp_delta[n_tokens x n_embd] = fc_out[n_tokens x hidden] @ proj_w[hidden x n_embd]
    mat_mul_float4(fc_out, proj_w, mlp_delta, n_tokens, n_embd, hidden);

    // Add proj bias
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < n_embd; r++) {
            mlp_delta[t * n_embd + r] += proj_b[r];
        }
    }

    free(fc_out);
}

float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x3)));
}
