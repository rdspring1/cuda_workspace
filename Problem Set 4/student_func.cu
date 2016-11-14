//Udacity HW 4
//Radix Sorting

#include <iostream>
#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const int THREADS = 128;
const int BLOCKS = 1723;
const int SCAN_THREADS = 512;

__global__
void
Scatter(unsigned int* const input,
        unsigned int* const inputPos,
        unsigned int* const output,
        unsigned int* const outputPos,
        unsigned int* const histogram,
        unsigned int* const zeros,
        unsigned int* const ones,
	const size_t numElems,
        const size_t bitshift)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numElems ) { return; }

    unsigned int digit = (input[idx] >> bitshift) & 0x1;
    unsigned int offset = (digit > 0) ? ones[idx] : zeros[idx];
    int outidx = histogram[digit] + offset;
    output[outidx] = input[idx];
    outputPos[outidx] = inputPos[idx];
}

__global__
void
Histogram(unsigned int* const input,
          unsigned int* const zeros,
          unsigned int* const ones,
	  const size_t numElems,
          const size_t bitshift,
	  unsigned int* const histogram)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numElems ) { return; }

    int digit = (input[idx] >> bitshift) & 0x1;
    zeros[idx] = (digit == 0);
    ones[idx] = (digit == 1);
    atomicAdd(&histogram[digit+1], 1);
}

__global__
void
Scan(unsigned int* const data, unsigned int* const max)
{
	extern __shared__ unsigned int temp[];
	int block_offset = blockIdx.x * 2 * blockDim.x;
	int tid = threadIdx.x;
	int offset = 1;

	temp[2*tid] = data[2*tid + block_offset];
	temp[2*tid+1] = data[2*tid+1 + block_offset];
	__syncthreads();

	for(int d = blockDim.x; d > 0; d >>= 1)
	{
		if(tid < d)
		{
			int l = offset * (2*tid+1) - 1;
			int r = offset * (2*tid+2) - 1;
			temp[r] += temp[l];
		}
		offset *= 2;
		__syncthreads();
	}

	if(tid == 0)
	{
		max[blockIdx.x] = temp[2*blockDim.x-1];
		temp[2*blockDim.x-1] = 0;
	}

	for(int d = 1; d < 2*blockDim.x; d <<= 1)
	{
		offset /= 2;
		__syncthreads();
		if(tid < d)
		{
			int l = offset * (2*tid+1) - 1;
			int r = offset * (2*tid+2) - 1;
			unsigned int t = temp[l];
			temp[l] = temp[r];
			temp[r] += t;
		}
	}
	__syncthreads();

	data[2*tid + block_offset] = temp[2*tid];
	data[2*tid+1 + block_offset] = temp[2*tid+1];
}

__global__
void
Add(unsigned int* const data, unsigned int* const offset)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	data[2*idx] += offset[blockIdx.x];
	data[2*idx+1] += offset[blockIdx.x];
}

void Scan(unsigned int* const data, const size_t size)
{
  const size_t SCAN_BLOCKS = size / (SCAN_THREADS * 2);
  unsigned int* max = NULL;
  checkCudaErrors(cudaMalloc(&max, SCAN_BLOCKS * sizeof(unsigned int)));
  Scan<<<SCAN_BLOCKS, SCAN_THREADS, SCAN_THREADS * 2 * sizeof(unsigned int)>>>(data, max);

  unsigned int* single_max = NULL;
  checkCudaErrors(cudaMalloc(&single_max, sizeof(unsigned int)));
  Scan<<<1, SCAN_THREADS, SCAN_THREADS * 2 * sizeof(unsigned int)>>>(max, single_max);

  Add<<<SCAN_BLOCKS, SCAN_THREADS>>>(data, max);
  cudaFree(max);
  cudaFree(single_max);
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  unsigned int* histogram = NULL;
  checkCudaErrors(cudaMalloc(&histogram, 3 * sizeof(unsigned int))); 

  unsigned int* zeros = NULL;
  unsigned int* ones = NULL;
  const size_t SCAN_BLOCKS = numElems / (SCAN_THREADS * 2) + 1;
  const size_t padded_size = SCAN_BLOCKS * SCAN_THREADS * 2;
  checkCudaErrors(cudaMalloc(&zeros, padded_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&ones, padded_size * sizeof(unsigned int)));

  for(size_t idx = 0; idx < 8 * sizeof(unsigned int); ++idx)
  {
     checkCudaErrors(cudaMemset(histogram, 0, 3 * sizeof(unsigned int)));

     if((idx % 2) == 0)
     {
     	Histogram<<<BLOCKS, THREADS>>>(d_inputVals, zeros, ones, numElems, idx, histogram);

     	Scan(zeros, padded_size);
     	Scan(ones, padded_size);
	Scatter<<<BLOCKS, THREADS>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, histogram, zeros, ones, numElems, idx);
     }
     else
     {
     	Histogram<<<BLOCKS, THREADS>>>(d_outputVals, zeros, ones, numElems, idx, histogram);

     	Scan(zeros, padded_size);
     	Scan(ones, padded_size);
     	Scatter<<<BLOCKS, THREADS>>>(d_outputVals, d_outputPos, d_inputVals, d_inputPos, histogram, zeros, ones, numElems, idx);
     }
  }
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  cudaFree(zeros);
  cudaFree(ones);
  cudaFree(histogram);
}
