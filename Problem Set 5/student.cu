/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <iostream>
#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals,
	       int numBins)
{
        extern __shared__ unsigned int temp[];
	temp[threadIdx.x] = 0;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= numVals) { return; }
        atomicAdd(&temp[vals[idx]], 1);
        __syncthreads();
        
        for(int pos = threadIdx.x; pos < numBins; pos+=blockDim.x)
        {
                atomicAdd(&histo[pos], temp[pos]);
        }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo, //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  const int THREADS = 1024;
  const int BLOCKS = numElems / THREADS;
  yourHisto<<<BLOCKS, THREADS, numBins * sizeof(unsigned int)>>>(d_vals, d_histo, numElems, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
