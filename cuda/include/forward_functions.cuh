#pragma once 
#include <cuda_runtime.h>

//forward kernels
__global__ void naive_gemm(const float* input_tensor, const float* weights, const float* bias, float* output, const int batch_dim, const int input_dim, const int output_dim);

__global__ void relu(float* input_tensor, const int batch_dim, const int input_dim);

__global__ void fused_gemm_relu(const float* input_tensor, const float* weights, const float* bias, float* output, const int batch_dim, const int input_dim, const int output_dim);

__global__ void softmax(float* input_tensor, float* output, const int batch_dim, const int input_dim);

// TODO: essentially a tiled matmul and softmax (much like flash attention)
//__global__ void fused_gemm_softmax(const float* input_tensor, const float* weights, const float* bias, float* output);
