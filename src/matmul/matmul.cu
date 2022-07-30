// Copyright (c) 2022 Milan Ondrasovic, milan.ondrasovic@gmail.com
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files(the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and / or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#ifndef TRUE
#define TRUE 1
#endif // TRUE

#ifndef FALSE
#define FALSE 0
#endif // FALSE

#define EPSILON 1e-4f

#define MATRIX_SIZE 4

__host__ int all_close(float *arr_a, float *arr_b, int len)
{
    for (int i = 0; i < len; ++i)
    {
        if (abs(arr_a[i] - arr_b[i]) > EPSILON)
        {
            return FALSE;
        }
    }

    return TRUE;
}

__device__ float rowcol_dot(
    float *matrix_a, float *matrix_b, int row, int col, int size)
{
    float sum = 0;

    for (int k = 0; k < size; ++k)
    {
        sum += matrix_a[row * size + k] * matrix_b[k * size + col];
    }

    return sum;
}

__global__ void matmul_ker(
    float *matrix_a, float *matrix_b, float *output, int size)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    output[row * size + col] = rowcol_dot(matrix_a, matrix_b, row, col, size);
}

__host__ int main()
{
    int n_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    float h_matrix_a[] = {
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0};

    float h_matrix_b[] = {14.0, 13.0, 12.0, 11.0,
                          14.0, 13.0, 12.0, 11.0,
                          14.0, 13.0, 12.0, 11.0,
                          14.0, 13.0, 12.0, 11.0};

    float h_matmul_output_gt[] = {140.0, 130.0, 120.0, 110.0,
                                  140.0, 130.0, 120.0, 110.0,
                                  140.0, 130.0, 120.0, 110.0,
                                  140.0, 130.0, 120.0, 110.0};

    float *d_matrix_a;
    float *d_matrix_b;
    float *d_matmul_output;

    cudaMalloc((float **)&d_matrix_a, n_bytes);
    cudaMalloc((float **)&d_matrix_b, n_bytes);
    cudaMalloc((float **)&d_matmul_output, n_bytes);

    cudaMemcpy(d_matrix_a, h_matrix_a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b, n_bytes, cudaMemcpyHostToDevice);

    float *h_matmul_output = (float *)malloc(n_bytes);

    dim3 block(2, 2, 1);
    dim3 grid(2, 2, 1);

    matmul_ker<<<grid, block>>>(
        d_matrix_a, d_matrix_b, d_matmul_output, MATRIX_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(
        h_matmul_output, d_matmul_output, n_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matmul_output);

    cudaDeviceReset();

    int status = all_close(
        h_matmul_output, h_matmul_output_gt, MATRIX_SIZE * MATRIX_SIZE);
    free(h_matmul_output);

    if (status)
    {
        printf("success: matrix multiplication outputs match\n");
    }
    else
    {
        fprintf(stderr, "error: matrix multiplication outputs differ\n");
    }

    return 1 - status;
}
