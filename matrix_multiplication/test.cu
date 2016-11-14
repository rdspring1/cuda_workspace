#include <math.h>
#include <stdio.h>
#include <random>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#include <cublas_v2.h>
#include <curand.h>

#include "utils.h"
#include "timer.h"

#define IDX2D(i,j, xdim, ydim) ((j)*(xdim)+(i)) 
#define THREADS_SIZE 32
//const int SHARED_ELEM = 4;
//const int SHARED_SIZE = SHARED_ELEM * SHARED_ELEM;
const float probability = 0.0625;
const int ACTIVE_ROWS = 64;
const int SHARED_ELEM = 32;
const int SHARED_SIZE = SHARED_ELEM;

void GPU_fill_rand(float *A, int rows, int cols) 
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, rows * cols);
}

void GPU_fill(float *A, int rows, int cols, int value) 
{
	for(int idx = 0; idx < rows; ++idx)
	{
		for(int jdx = 0; jdx < cols; ++jdx)
		{
			A[idx * rows + jdx] = jdx % value;
		}
	}
}

void GPU_fill_static(float *A, int rows, int cols, int value) 
{
	for(int idx = 0; idx < rows; ++idx)
	{
		for(int jdx = 0; jdx < cols; ++jdx)
		{
			A[idx * rows + jdx] = value;
		}
	}
}

void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n)
{
	const float alpha = 1;
	const float beta = 0;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
}

	__global__
void inner_multiply(const float *A, const float *B, float *C, int rows, int k, int cols)
{
	float value[SHARED_SIZE];
	for(int offset = 0; offset < SHARED_SIZE; ++offset)
	{
		value[offset] = 0;
	}

	for(int kdx = threadIdx.x; kdx < k; kdx+=blockDim.x)
	{
		for(int offset = 0; offset < SHARED_SIZE; ++offset)
		{
			int idx = offset / SHARED_ELEM;
			int jdx = offset % SHARED_ELEM;
			value[offset] += A[IDX2D(kdx, blockIdx.x + idx, rows, k)] * B[IDX2D(kdx, blockIdx.y + jdx, k, cols)];
		}
	}
	__syncthreads();

	// Warp Shuffle = Reduction -> Dot Product
	for(int offset = 0; offset < SHARED_SIZE; ++offset)
	{
		value[offset] += __shfl_down(value[offset], 16);
		value[offset] += __shfl_down(value[offset], 8);
		value[offset] += __shfl_down(value[offset], 4);
		value[offset] += __shfl_down(value[offset], 2);
		value[offset] += __shfl_down(value[offset], 1);
	}

	if (threadIdx.x == 0)
	{
		for(int offset = 0; offset < SHARED_SIZE; ++offset)
		{
			int idx = offset / SHARED_ELEM;
			int jdx = offset % SHARED_ELEM;
			C[IDX2D(blockIdx.x * SHARED_ELEM + idx, blockIdx.y * SHARED_ELEM + jdx, rows, cols)] = value[offset];
		}
	}
}

	__global__
void lsh_multiply(const int* active, const float *A, const float *B, float *C, int rows, int k, int cols)
{
	int indices[SHARED_SIZE];
	float value[SHARED_SIZE];
	int global_offset = blockIdx.x * SHARED_SIZE;
	for(int offset = 0; offset < SHARED_SIZE; ++offset)
	{
		value[offset] = 0;
		indices[offset] = active[global_offset + offset];
	}

	for(int kdx = threadIdx.x; kdx < k; kdx+=blockDim.x)
	{
		for(int offset = 0; offset < SHARED_SIZE; ++offset)
		{
			value[offset] += A[IDX2D(kdx, indices[offset], rows, k)] * B[IDX2D(kdx, blockIdx.y, k, cols)];
		}
	}
	__syncthreads();

	// Warp Shuffle = Reduction -> Dot Product
	for(int offset = 0; offset < SHARED_SIZE; ++offset)
	{
		value[offset] += __shfl_down(value[offset], 16);
		value[offset] += __shfl_down(value[offset], 8);
		value[offset] += __shfl_down(value[offset], 4);
		value[offset] += __shfl_down(value[offset], 2);
		value[offset] += __shfl_down(value[offset], 1);
	}

	if (threadIdx.x == 0)
	{
		for(int offset = 0; offset < SHARED_SIZE; ++offset)
		{
			C[IDX2D(indices[offset], blockIdx.y, rows, cols)] = value[offset];
		}
	}
}

	__global__
void multiply(const float *A, const float *B, float *C, int rows, int k, int cols, int x, int y)
{
	float value[SHARED_SIZE];
	for(int offset = 0; offset < SHARED_SIZE; ++offset)
	{
		value[offset] = 0;
	}
	for(int kdx = threadIdx.x; kdx < k; kdx+=blockDim.x)
	{
		for(int offset = 0; offset < SHARED_SIZE; ++offset)
		{
			value[offset] += A[IDX2D(kdx, x + offset, rows, k)] * B[IDX2D(kdx, y, k, cols)];
		}
	}
	__syncthreads();

	// Warp Shuffle = Reduction -> Dot Product
	for(int offset = 0; offset < SHARED_SIZE; ++offset)
	{
		value[offset] += __shfl_down(value[offset], 16);
		value[offset] += __shfl_down(value[offset], 8);
		value[offset] += __shfl_down(value[offset], 4);
		value[offset] += __shfl_down(value[offset], 2);
		value[offset] += __shfl_down(value[offset], 1);
	}

	if (threadIdx.x == 0)
	{
		for(int offset = 0; offset < SHARED_SIZE; ++offset)
		{
			C[IDX2D(x * SHARED_ELEM + offset, y, rows, cols)] = value[offset];
		}
	}
}

int main()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	const int rows = 1024;
	const int k = 1024;
	const int cols = 1024;

	float* A = NULL;
	float* B = NULL;
	float* C = NULL;
	float* my_C = NULL;
	checkCudaErrors(cudaMallocManaged(&A, sizeof(float) * rows * k));
	checkCudaErrors(cudaMallocManaged(&B, sizeof(float) * k * cols));
	checkCudaErrors(cudaMallocManaged(&C, sizeof(float) * rows * cols));
	checkCudaErrors(cudaMallocManaged(&my_C, sizeof(float) * rows * cols));

	GPU_fill_static(A, rows, k, 10000);
	GPU_fill_static(B, k, cols, 10000);

        const int num_streams = 16;
	cudaStream_t streams[num_streams];
	for(int idx = 0; idx < num_streams; ++idx)
	{
		cudaStreamCreate(&streams[idx]);
	}

	int* active = NULL;
	checkCudaErrors(cudaMallocManaged(&active, sizeof(int) * ACTIVE_ROWS * cols));
	checkCudaErrors(cudaMemset(active, 0, sizeof(int) * ACTIVE_ROWS * cols));
	int count = -1;
	for(int idx = 0; idx < rows * cols; ++idx)
	{
		if(distribution(generator) < probability)
		{
			active[++count] = idx % rows;
		}
	}
	//printf("Count: %d\n", count);

	// Add Data and Query
	GpuTimer timer;
	timer.Start();

	// CUBLAS
	//gpu_blas_mmul(handle, A, B, C, rows, k, cols);
	//checkCudaErrors(cudaDeviceSynchronize());

	// My_SGEMM 
	//const dim3 gridSize(rows/SHARED_ELEM, cols/SHARED_ELEM, 1);
	//const dim3 blockSize(THREADS_SIZE, 1, 1);
	//inner_multiply<<<gridSize, blockSize>>>(A, B, my_C, rows, k, cols);
	//checkCudaErrors(cudaDeviceSynchronize());

	// My_LSH_SGEMM 
	const dim3 gridSize(ACTIVE_ROWS/SHARED_SIZE, cols, 1);
	const dim3 blockSize(THREADS_SIZE, 1, 1);
	lsh_multiply<<<gridSize, blockSize>>>(active, A, B, my_C, rows, k, cols);
	checkCudaErrors(cudaDeviceSynchronize());

	// Asynchronous Inner Product
	/*
	for(int idx = 0; idx < rows/SHARED_ELEM; ++idx)
	{
		for(int jdx = 0; jdx < cols; ++jdx)
		{
			int pos = IDX2D(idx, jdx, rows, cols) % num_streams;
			multiply<<<1, blockSize, 0, streams[pos]>>>(A, B, my_C, rows, k, cols, idx, jdx);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());
	*/
	timer.Stop();

	//checkResultsExact(C, my_C, rows * cols);
	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
	if (err < 0)
	{
		//Couldn't print! Probably the student closed stdout - bad news
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}

	// Free GPU memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(my_C);

	// Destroy the handle
	cublasDestroy(handle);

	for(int idx = 0; idx < num_streams; ++idx)
	{
		cudaStreamDestroy(streams[idx]);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	return 0;
}
