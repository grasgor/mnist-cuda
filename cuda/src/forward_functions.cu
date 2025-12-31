#include <cuda_runtime.h>
#include "forward_functions.cuh"


__global__ void naive_gemm(const float* input_tensor,
							const float* weights,
	 						const float* bias,
							const int batch_dim,
							const int input_dim,
							const int output_dim,
							float* output){
	//input tensor is of dim (batch_dim, 1, 784)
	//we'll still launch a 2D grid since the matrix stored in row-major
	//behaves as (batch_dim, 784) effectively

	const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

	const int M = batch_dim;
	const int K = input_dim;
	const int N = output_dim;

	if (row < M && col < N){
		float temp = 0.0f;

		#pragma unroll
		for(int k = 0; k<K; k++){
			temp = fmaf(input_tensor[row * K + k], weights[k * N + col], temp);
		}
		temp += bias[col];
		output[row * N + col] = temp; 
}
}

__global__ void relu(float* input_tensor, const int batch_dim, const int input_dim) {

    const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    float4* input_tensor4 = reinterpret_cast<float4*>(input_tensor);

    int col4 = input_dim / 4;   
    int tail_rem = input_dim % 4; 

    if (row < batch_dim && col < col4) {
        int idx4 = row * col4 + col;
        float4 u = input_tensor4[idx4];

        u.x = fmaxf(0.0f, u.x);
        u.y = fmaxf(0.0f, u.y);
        u.z = fmaxf(0.0f, u.z);
        u.w = fmaxf(0.0f, u.w);

        input_tensor4[idx4] = u;
    }

    if (tail_rem > 0 && row < batch_dim && col < tail_rem) {
        int base = row * input_dim + col4 * 4;
        int idx = base + col;

        input_tensor[idx] = fmaxf(0.0f, input_tensor[idx]);
    }
}


__global__ void fused_gemm_relu(const float* input_tensor, 
								const float* weights,
								const float* bias,
								float* output,
								const int batch_dim,
								const int input_dim,
								const int output_dim){
	//input tensor is of dim (batch_dim, 1, 784)
	//we'll still launch a 2D grid since the matrix stored in row-major
	//behaves as (batch_dim, 784) effectively

	const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

	const int M = batch_dim;
	const int K = input_dim;
	const int N = output_dim;

	if (row < M && col < N){
		float temp = 0.0f;

		#pragma unroll
		for(int k = 0; k<K; k++){
			temp = fmaf(input_tensor[row * K + k], weights[k * N + col], temp);
		}
		temp += bias[col]; // add bias
		output[row * N + col] = fmaxf(0.0f, temp); // apply relu
}
}

__global__ void softmax(float* input_tensor,
						float* output_tensor,
						const int batch_dim,
						const int input_dim){
	//online softmax that uses block level reductions

	//dynamic allocation of shared memory
	extern __shared__ float smem[];

	//let the blockId attend to rows
	const unsigned int row = blockIdx.x;
	const unsigned int tid_x = threadIdx.x;

	if (row >= batch_dim) return;
	//pointer arithemtic to move pointer row by row
	float* dinput = input_tensor + row * input_dim;  
	float* doutput = output_tensor + row * input_dim;

	/**
	core logic for online softmax
	each thread accesses row elements in a strided fashion, this set of elements has a local max and local norm
	the local max and local norm are calculated based on the online softmax principle with norm error correction term
	**/

	//local_max and local_norm are thread specific register variables
	float local_max = -INFINITY;
	float local_norm = 0.0f;
	for(int i = tid_x; i < input_dim; i += blockDim.x){
		float curr_x = dinput[i];
		if(curr_x > local_max){
			local_norm *= __expf(local_max - curr_x);
			local_max = curr_x;
		}
		local_norm += __expf(curr_x - local_max);
	}

	/**
	For the corresponding tid_x having calculated the local max, we store it in smem for reduction later 
	**/							
	smem[tid_x] = local_max;

	//sync threads before proceeding
	__syncthreads();

	//before block reduction to achieve row_max, reduction performed in logn
	for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
		if(tid_x < stride){
			smem[tid_x] = fmaxf(smem[tid_x], smem[tid_x + stride]);
		}
		__syncthreads();
	}
	//first element is now the row max after reduction
	float row_max = smem[0];

	//reuse the same shared memory to now load local norm from thread registers while simultaneously performing error correction
	smem[tid_x] = local_norm * __expf(local_max - row_max);
	__syncthreads();

	//perform addition reduction
	for(int stride = blockDim.x / 2; stride > 0 ; stride /= 2){
		if(tid_x < stride){
			smem[tid_x] += smem[tid_x + stride];
		}
		__syncthreads();
	}

	/**
	after reductions on our first pass of the row
	we have the row_max and row_norm
	**/
	float row_norm = smem[0];

	//second pass to actually calculate softmax for the row
	for(int idx = tid_x; idx < input_dim; idx += blockDim.x){
		doutput[idx] = __expf(dinput[idx] - row_max) / row_norm;
	}
}