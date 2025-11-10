#pragma once
#include <cstdint>

void one_hot_encode(const int64_t* targets, float* one_hot_targets, const int batch_dim, const int output_dim);

float cross_entropy_loss(const float* one_hot_targets, const float* logits,
                         int batch_dim, int output_dim);
void calculate_grad(
    const float* X, const float* logits, const float* one_hot_targets,
    float* Z1, float* A1, const int batch_dim, const int input_dim,
    const int hidden_dim, const int output_dim,
    float* W1, float* W2,
    float* dW2, float* db2, float* dW1, float* db1, float batch_loss);

