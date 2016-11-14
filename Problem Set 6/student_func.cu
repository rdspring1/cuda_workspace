//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>

#define THREADS 32

__global__
void maskIndex(const uchar4* source, int* mask, const size_t rows, const size_t cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x * cols + y;
	if( idx >= rows * cols )
	{
		return;
	}

	mask[idx] = (source[idx].x < 255) & (source[idx].y < 255) & (source[idx].z < 255);
}

__global__
void borderIndex(int* mask, int* border, const size_t rows, const size_t cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if( x >= rows-1 || y >= cols-1 )
	{
		return;
	}

	int idx = x * cols + y;
	int top = (x-1) * cols + y;
	int bottom = (x+1) * cols + y;
	int left = x * cols + (y-1);
	int right = x * cols + (y+1);
	border[idx] = !(mask[top] & mask[bottom] & mask[left] & mask[right]) & mask[idx]; 
}

__global__
void StructToArray(const uchar4* source, float* red, float* green, float* blue, const size_t rows, const size_t cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x * cols + y;
	if( idx >= rows * cols )
	{
		return;
	}

	red[idx] = (float) source[idx].x;
	green[idx] = (float) source[idx].y;
	blue[idx] = (float) source[idx].z;
}

__global__
void ArrayToStruct(uchar4* dest, float* red, float* green, float* blue, const size_t rows, const size_t cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x * cols + y;
	if( idx >= rows * cols )
	{
		return;
	}

	dest[idx].x = red[idx];
	dest[idx].y = green[idx];
	dest[idx].z = blue[idx];
}

__host__ __device__ 
void swap(float** left, float** right)
{
	float* temp = *right;
	*right = *left;
	*left = temp;
}

__global__
void jacobi(float* prev, float* next, int* mask, int* border, float* source, float* dest, const int iterations, const size_t rows, const size_t cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x * cols + y;
	if( idx >= rows * cols || !mask[idx] )
	{
		return;
	}

	int top = (x-1) * cols + y;
	int bottom = (x+1) * cols + y;
	int left = x * cols + (y-1);
	int right = x * cols + (y+1);

	bool topBorder = border[top];
	bool bottomBorder = border[bottom];
	bool leftBorder = border[left];
	bool rightBorder = border[right];

	float topValue = dest[top];
	float bottomValue = dest[bottom];
	float leftValue = dest[left];
	float rightValue = dest[right];
	
	float sum2 = 0;
      	sum2 += source[idx] - source[top];
      	sum2 += source[idx] - source[bottom];
      	sum2 += source[idx] - source[left];
      	sum2 += source[idx] - source[right];

	float sum1 = 0;
	sum1 += (topBorder) ? topValue : prev[top];
	sum1 += (bottomBorder) ? bottomValue : prev[bottom];
	sum1 += (leftBorder) ? leftValue : prev[left];
	sum1 += (rightBorder) ? rightValue : prev[right];

      	float newVal = (sum1 + sum2) / 4.0f;
      	next[idx] = ::fmin(255.0f, ::fmax(0.0f, newVal)); //clamp to [0, 255]
}

__global__
void blend(float* blend, int* mask, int* border, float* value, const size_t rows, const size_t cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x * cols + y;
	if( idx >= rows * cols )
	{
		return;
	}

	if(mask[idx] & !border[idx])
	{
		blend[idx] = value[idx];
	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
 	const dim3 blockSize(THREADS, THREADS, 1);
  	const dim3 gridSize(numRowsSource/THREADS+1, numColsSource/THREADS+1, 1);

	uchar4* d_sourceImg = NULL;
	checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));	

	uchar4* d_destImg = NULL;
	checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));

	// Part 1
	int* mask = NULL;
	checkCudaErrors(cudaMalloc(&mask, sizeof(int) * numRowsSource * numColsSource));
	maskIndex<<<gridSize, blockSize>>>(d_sourceImg, mask, numRowsSource, numColsSource);

	// Part 2
	int* border = NULL;
	checkCudaErrors(cudaMalloc(&border, sizeof(int) * numRowsSource * numColsSource));
	borderIndex<<<gridSize, blockSize>>>(mask, border, numRowsSource, numColsSource);

	// Part 3
	float* d_red_source = NULL;
	checkCudaErrors(cudaMalloc(&d_red_source, sizeof(float) * numRowsSource * numColsSource));
	float* d_green_source = NULL;
	checkCudaErrors(cudaMalloc(&d_green_source, sizeof(float) * numRowsSource * numColsSource));
	float* d_blue_source = NULL;
	checkCudaErrors(cudaMalloc(&d_blue_source, sizeof(float) * numRowsSource * numColsSource));
	StructToArray<<<gridSize, blockSize>>>(d_sourceImg, d_red_source, d_green_source, d_blue_source, numRowsSource, numColsSource);

	float* d_red_dest = NULL;
	checkCudaErrors(cudaMalloc(&d_red_dest, sizeof(float) * numRowsSource * numColsSource));
	float* d_green_dest = NULL;
	checkCudaErrors(cudaMalloc(&d_green_dest, sizeof(float) * numRowsSource * numColsSource));
	float* d_blue_dest = NULL;
	checkCudaErrors(cudaMalloc(&d_blue_dest, sizeof(float) * numRowsSource * numColsSource));
	StructToArray<<<gridSize, blockSize>>>(d_destImg, d_red_dest, d_green_dest, d_blue_dest, numRowsSource, numColsSource);

	// Part 4
	const int iterations = 800;
	float* prev = NULL;
	checkCudaErrors(cudaMalloc(&prev, sizeof(float) * numRowsSource * numColsSource));
	float* next = NULL;
	checkCudaErrors(cudaMalloc(&next, sizeof(float) * numRowsSource * numColsSource));

	// Part 5 + 6
	float* d_red_blend = NULL;
	float* d_green_blend = NULL;
	float* d_blue_blend = NULL;
	checkCudaErrors(cudaMalloc(&d_red_blend, sizeof(float) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_green_blend, sizeof(float) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_blue_blend, sizeof(float) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_red_blend, d_red_dest, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_green_blend, d_green_dest, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_blue_blend, d_blue_dest, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpy(prev, d_red_source, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
	for(int rnd = 0; rnd < iterations; ++rnd)
	{
		jacobi<<<gridSize, blockSize>>>(prev, next, mask, border, d_red_source, d_red_dest, iterations, numRowsSource, numColsSource);
		swap(&prev, &next);
        	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	blend<<<gridSize, blockSize>>>(d_red_blend, mask, border, prev, numRowsSource, numColsSource);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(prev, d_green_source, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
	for(int rnd = 0; rnd < iterations; ++rnd)
	{
		jacobi<<<gridSize, blockSize>>>(prev, next, mask, border, d_green_source, d_green_dest, iterations, numRowsSource, numColsSource);
		swap(&prev, &next);
        	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	blend<<<gridSize, blockSize>>>(d_green_blend, mask, border, prev, numRowsSource, numColsSource);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(prev, d_blue_source, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
	for(int rnd = 0; rnd < iterations; ++rnd)
	{
		jacobi<<<gridSize, blockSize>>>(prev, next, mask, border, d_blue_source, d_blue_dest, iterations, numRowsSource, numColsSource);
		swap(&prev, &next);
        	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	blend<<<gridSize, blockSize>>>(d_blue_blend, mask, border, prev, numRowsSource, numColsSource);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	

	// Part 6
	uchar4* d_blendedImg = NULL;
	checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource));
	ArrayToStruct<<<gridSize, blockSize>>>(d_blendedImg, d_red_blend, d_green_blend, d_blue_blend, numRowsSource, numColsSource);
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyDeviceToHost));
	
	// Cleanup
	checkCudaErrors(cudaFree(d_sourceImg));
	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(d_blendedImg));
	checkCudaErrors(cudaFree(mask));
	checkCudaErrors(cudaFree(border));
	checkCudaErrors(cudaFree(prev));
	checkCudaErrors(cudaFree(next));
	checkCudaErrors(cudaFree(d_red_source));
	checkCudaErrors(cudaFree(d_green_source));
	checkCudaErrors(cudaFree(d_blue_source));
	checkCudaErrors(cudaFree(d_red_dest));
	checkCudaErrors(cudaFree(d_green_dest));
	checkCudaErrors(cudaFree(d_blue_dest));
	checkCudaErrors(cudaFree(d_red_blend));
	checkCudaErrors(cudaFree(d_green_blend));
	checkCudaErrors(cudaFree(d_blue_blend));
}
