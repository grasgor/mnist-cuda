#include <iostream>
#include <cmath>
#include <cstring>
#include "forward_functions.h"
#include "backward_functions.h"

void one_hot_encode(const int64_t* targets, float* one_hot_targets, const int batch_dim, const int output_dim){
	// targets is of shape (batch_dim, 1)
	// target labels go from 1-10 for MNIST
	
	//we use the same memory allocation and refresh it for each batch
	//set iall to zero using memset
	memset(one_hot_targets, 0, batch_dim * size_t(output_dim) * sizeof(float));
    for (int b = 0; b < batch_dim; ++b) {
        int64_t t = targets[b];
        if (t >= 0 && t < output_dim) {
            one_hot_targets[b * output_dim + static_cast<int>(t)] = 1.0f;
        }
    }
}

float cross_entropy_loss(const float* one_hot_targets, const float* logits,
                         int batch_dim, int output_dim) {
    float loss = 0.0f;
    const float epsilon = 1e-9f;

    for (int i = 0; i < batch_dim * output_dim; ++i) {
        if (one_hot_targets[i] > 0.0f)
            loss -= logf(logits[i] + epsilon);
    }
    return loss / float(batch_dim);
}




void calculate_grad(
    const float* X, const float* logits, const float* one_hot_targets,
    float* Z1, float* A1, const int batch_dim, const int input_dim,
    const int hidden_dim, const int output_dim,
    float* W1, float* W2,
    float* dW2, float* db2, float* dW1, float* db1, float batch_loss)
{

    float loss = cross_entropy_loss(one_hot_targets, logits, batch_dim, output_dim);
    batch_loss = loss;


    // dZ2 = logits - targets
    float* dZ2 = new float[batch_dim * output_dim];
    for(int b = 0; b < batch_dim; b++){
        for(int k = 0; k < output_dim; k++){
            int idx = b*output_dim + k;
            dZ2[idx] = logits[idx] - one_hot_targets[idx];
        }
    }

    // db2 = sum over batch
    for(int k = 0; k < output_dim; k++){
        float sum = 0.0f;
        for(int b = 0; b < batch_dim; b++){
            sum += dZ2[b*output_dim + k];
        }
        db2[k] = sum;
    }

    // dW2 = A1^T @ dZ2
    float* A1_T = new float[hidden_dim * batch_dim];
    transpose(A1, A1_T, 1, batch_dim, hidden_dim); // (H x B)
    matmul(A1_T, dZ2, dW2, 1, hidden_dim, batch_dim, output_dim); // (H x output_dim)

    // dZ1 = dZ2 @ W2^T .* ReLU'(Z1)
    float* W2_T = new float[output_dim * hidden_dim];
    transpose(W2, W2_T, 1, hidden_dim, output_dim); // (output_dim x H)

    float* dZ1 = new float[batch_dim * hidden_dim];
    matmul(dZ2, W2_T, dZ1, batch_dim, 1, output_dim, hidden_dim); // (B x H)

    // apply ReLU mask
    for(int i = 0; i < batch_dim * hidden_dim; i++){
        dZ1[i] *= (A1[i] > 0) ? 1.0f : 0.0f;
    }

    // db1 = sum over batch
    for(int h = 0; h < hidden_dim; h++){
        float sum = 0.0f;
        for(int b = 0; b < batch_dim; b++){
            sum += dZ1[b*hidden_dim + h];
        }
        db1[h] = sum;
    }

    // dW1 = X^T @ dZ1
    float* X_T = new float[input_dim * batch_dim];
    transpose(X, X_T, 1, batch_dim, input_dim); // (N x B)
    matmul(X_T, dZ1, dW1, 1, input_dim, batch_dim, hidden_dim); // (N x H)

    // free temporary arrays
    delete[] dZ2;
    delete[] A1_T;
    delete[] W2_T;
    delete[] dZ1;
    delete[] X_T;
}

