/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <iostream>

const int THREADS_PER_BLOCK = 128;

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__
void
MinMaxKernel(const float* const d_logLuminance, 
	     const size_t numRows,
	     const size_t numCols,
	     float* max,
	     float* min)
{
    __shared__ float smax[THREADS_PER_BLOCK];
    __shared__ float smin[THREADS_PER_BLOCK];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numCols * numRows ) { return; }
    smax[tid] = d_logLuminance[idx];
    smin[tid] = d_logLuminance[idx];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s/=2)
    {
        if (tid < s)
        {
           smax[tid] = ::fmaxf(smax[tid], smax[tid+s]);
           smin[tid] = ::fminf(smin[tid], smin[tid+s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicMax(max, smax[tid]);
        atomicMin(min, smin[tid]);
    }
}

__global__
void
Histogram(const float* const d_logLuminance,
	  unsigned int* const d_cdf,
	  float min,
	  float range,
	  const size_t numRows,
	  const size_t numCols,
	  const size_t numBins)
{
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numCols * numRows ) { return; }
    sdata[tid] = d_logLuminance[idx];
    __syncthreads();

    int bin = (int) ::fminf(numBins-1, (sdata[tid] - min) / range * numBins);
    atomicAdd(&d_cdf[bin], 1);
}

__global__
void
Scan(unsigned int* const data)
{
	extern __shared__ unsigned int temp[];
	int tid = threadIdx.x;
	int offset = 1;
	temp[2*tid] = data[2*tid];
	temp[2*tid+1] = data[2*tid+1];
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

	data[2*tid] = temp[2*tid];
	data[2*tid+1] = temp[2*tid+1];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*
   * Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)
    */

    float* max = NULL;
    float* min = NULL;
    checkCudaErrors(cudaMalloc(&max, sizeof(float)));
    checkCudaErrors(cudaMalloc(&min, sizeof(float)));

    std::cout << (numRows * numCols) << std::endl;
    MinMaxKernel<<<1024, THREADS_PER_BLOCK>>>(d_logLuminance, numRows, numCols, max, min);

    checkCudaErrors(cudaMemcpy(&max_logLum, max, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&min_logLum, min, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(max);
    cudaFree(min);

    float range = max_logLum - min_logLum;
    //std::cout << range << " " << max_logLum << " " << min_logLum << std::endl;
    Histogram<<<1024, THREADS_PER_BLOCK, sizeof(float) * THREADS_PER_BLOCK>>>(d_logLuminance, d_cdf, min_logLum, range, numRows, numCols, numBins);

    Scan<<<1, numBins/2, sizeof(unsigned int) * numBins>>>(d_cdf);
}
