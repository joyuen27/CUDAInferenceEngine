#include "llm_mat_mul.h"

#include <cuda_runtime.h>

#define TILE_SIZE 16
#define REG_TILE_SIZE 8

// C = A @ B,  A:[M x K], B:[K x N], C:[M x N], all row-major.
// Bounds-checked so M, N, K need not be multiples of TILE_SIZE*REG_TILE_SIZE.
__global__ void k_mat_mul_float4(const float* dev_a, const float* dev_b, float* dev_c,
                                 int a_num_rows, int b_num_cols, int dim_shared) {
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE * REG_TILE_SIZE]; // Transposed so we can get register tile in contiguous memory
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE * REG_TILE_SIZE];

    int c_row = (blockIdx.y * TILE_SIZE + threadIdx.y) * REG_TILE_SIZE;
    int c_col = (blockIdx.x * TILE_SIZE + threadIdx.x) * REG_TILE_SIZE;

    int num_tile_steps = (dim_shared + TILE_SIZE - 1) / TILE_SIZE;

    float accum[REG_TILE_SIZE][REG_TILE_SIZE] = {};

    for (int t = 0; t < num_tile_steps; t++)
    {
        //Determine row/col for A and B
        int a_row = (blockIdx.y * TILE_SIZE + threadIdx.y) * REG_TILE_SIZE;
        int a_col = TILE_SIZE * t + threadIdx.x;
        int b_row = TILE_SIZE * t + threadIdx.y;
        int b_col = (blockIdx.x * TILE_SIZE + threadIdx.x) * REG_TILE_SIZE;

        //Pull global memory into shared (zero-fill out-of-range so padded lanes contribute 0)
        for (int i = 0; i < REG_TILE_SIZE; i++) {
            float a_val = 0.0f;
            if ((a_row + i) < a_num_rows && a_col < dim_shared) {
                a_val = dev_a[(a_row + i) * dim_shared + a_col];
            }
            a_shared[threadIdx.x][threadIdx.y * REG_TILE_SIZE + i] = a_val;

            float b_val = 0.0f;
            if (b_row < dim_shared && (b_col + i) < b_num_cols) {
                b_val = dev_b[b_row * b_num_cols + b_col + i];
            }
            b_shared[threadIdx.y][threadIdx.x * REG_TILE_SIZE + i] = b_val;
        }

        //Synchronize all threads in block
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
        {
            float a_reg[REG_TILE_SIZE];
            float b_reg[REG_TILE_SIZE];

            for (int j = 0; j < REG_TILE_SIZE; j += 4)
            {
                *reinterpret_cast<float4*> (&a_reg[j]) = *reinterpret_cast<float4*> (&a_shared[i][threadIdx.y * REG_TILE_SIZE + j]);
                *reinterpret_cast<float4*> (&b_reg[j]) = *reinterpret_cast<float4*> (&b_shared[i][threadIdx.x * REG_TILE_SIZE + j]);
            }

            for (int r = 0; r < REG_TILE_SIZE; r++) {
                for (int c = 0; c < REG_TILE_SIZE; c++) {
                    accum[r][c] += a_reg[r] * b_reg[c];
                }
            }
        }
        //Synchronize all threads in block
        __syncthreads();
    }

    //Set output elems (bounds-checked)
    for (int r = 0; r < REG_TILE_SIZE; r++) {
        for (int c = 0; c < REG_TILE_SIZE; c++) {
            int row = c_row + r;
            int col = c_col + c;
            if (row < a_num_rows && col < b_num_cols) {
                dev_c[row * b_num_cols + col] = accum[r][c];
            }
        }
    }
}

void mat_mul_float4(const float* a, const float* b, float* c, int M, int N, int K) {
    //Set up device pointers
    float* dev_a;
    float* dev_b;
    float* dev_c;

    //Allocate on device
    cudaMalloc((void**) &dev_a, (size_t) M * K * sizeof(float));
    cudaMalloc((void**) &dev_b, (size_t) K * N * sizeof(float));
    cudaMalloc((void**) &dev_c, (size_t) M * N * sizeof(float));

    //Copy a and b to device
    cudaMemcpy(dev_a, a, (size_t) M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, (size_t) K * N * sizeof(float), cudaMemcpyHostToDevice);

    //Define grid and block dimensions
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE * REG_TILE_SIZE - 1) / (TILE_SIZE * REG_TILE_SIZE),
                  (M + TILE_SIZE * REG_TILE_SIZE - 1) / (TILE_SIZE * REG_TILE_SIZE));

    //Call kernel
    k_mat_mul_float4<<<grid_dim, block_dim>>>(dev_a, dev_b, dev_c, M, N, K);

    //Synchronize
    cudaDeviceSynchronize();

    //Copy output matrix back
    cudaMemcpy(c, dev_c, (size_t) M * N * sizeof(float), cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
