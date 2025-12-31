#include <cmath>
#include <random>
#include "forward_functions.h"
using namespace std;

void matmul(const float* input, const float* weights, float* output, const int batch_dim, const int m, const int n, const int output_dim){
	//input is of shape (B, M, N)
	//weight is of shape (N, output_dim)
	for(int b = 0; b<batch_dim; b++){
		for(int y = 0; y < m; y++){
			for(int k = 0; k <output_dim; k++){
				float temp = 0.0f;
				for(int x = 0; x<n; x++){
					temp += input[b*m*n + y*n + x] * weights[x*output_dim + k]; 
				}
				output[b*m*output_dim + y * output_dim + k] = temp;
			}
		}
	}
} 



void transpose(const float* input, float* output, const int batch_dim, const int M, const int N){
	for(int b = 0; b<batch_dim; b++){
		for(int y = 0; y<M; y++){
			for(int x = 0; x<N; x++){
				int idx = b*M*N + y*N + x;
				int transpose_idx = b*M*N + x*M + y;
				output[transpose_idx] = input[idx];
			}
		}
	}
}


void relu(const float *Z1, float* A1, const int batch_dim, const int m, const int n){
	for(int i = 0; i < batch_dim * m * n; i++){
		A1[i] = max(0.0f, Z1[i]);
	}
}

void softmax(const float *Z2, float* output, const int batch_dim, const int m, const int n){

	for(int b = 0; b < batch_dim; b++){
		for(int y = 0; y<m; y++){
			float row_max = -INFINITY;
			for(int x = 0; x<n; x++){
				float val = Z2[b*m*n + y*n +x]; 
				if(val > row_max) {
					row_max = val;
				}
			}
			float exp_sum = 0.0f;
            		for (int x = 0; x < n; x++) {
                		float e_val = expf(Z2[b * m * n + y * n + x] - row_max);
                		output[b * m * n + y * n + x] = e_val;
                		exp_sum += e_val;
            		}
			for(int x = 0; x<n; x++){
				output[b*m*n + y*n + x] /= exp_sum; 
			}
		}
	}
}


void generate_image_batch(float *input, const int batch_dim, const int image_dim){

	mt19937 gen(42); //
	uniform_real_distribution<float> dist(0.0f, 1.0f);
	for(int b = 0; b<batch_dim; b++){
		for(int i = 0; i<image_dim; i++){
			input[image_dim*b + i] = dist(gen);
		}
	}
}


void xavier_weight_init(float* weights, const int fan_in, const int fan_out){

	mt19937 gen(42); //random seed set to 42
	float bound = sqrt(6.0f/ (fan_in + fan_out)); // abs bound for xavier weights
	uniform_real_distribution<float> dist(-bound, bound);
	for(int i = 0; i < fan_in*fan_out; i++){
		weights[i] = dist(gen);
	}
}



