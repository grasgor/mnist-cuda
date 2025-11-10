#pragma once

void matmul(const float* input, const float* weights, float* output, const int batch_dim, const int M, const int N, const int K);

void transpose(const float* input, float* output, const int batch_dim, const int M, const int N);

void relu(const float *Z1, float* output, const int batch_dim, const int M, const int N);

void softmax(const float *Z2, float* output, const int batch_dim, const int M, const int N);

void generate_image_batch(float *input, const int batch_dim = 16, const int image_dim = 784);

void xavier_weight_init(float* weights, const int fan_in, const int fan_out);
