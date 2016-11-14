// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

	__global__
void gaussian_blur(const unsigned char* const inputChannel,
		unsigned char* const outputChannel,
		int numRows, int numCols,
		const float* const filter, const int filterWidth)
{
	const int width = blockDim.x + filterWidth;
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y * 2 + threadIdx.y;
	if ( X >= numCols || Y >= numRows ) { return; }

	extern __shared__ float data[];
	data[(threadIdx.y + filterWidth/2) * width + (threadIdx.x + filterWidth/2)] = inputChannel[Y * numCols + X];
	data[(threadIdx.y + blockDim.y + filterWidth/2) * width + (threadIdx.x + filterWidth/2)] = inputChannel[(Y + blockDim.y) * numCols + X];

	if(threadIdx.x < filterWidth/2)
	{
		data[(threadIdx.y + filterWidth/2) * width + threadIdx.x] = inputChannel[Y * numCols + X - filterWidth/2];
		data[(threadIdx.y + blockDim.y + filterWidth/2) * width + threadIdx.x] = inputChannel[(Y + blockDim.y) * numCols + X - filterWidth/2];
		data[(threadIdx.y + filterWidth/2) * width + (threadIdx.x + filterWidth/2 + blockDim.x)] = inputChannel[Y * numCols + X + blockDim.x];
		data[(threadIdx.y + blockDim.y + filterWidth/2) * width + (threadIdx.x + filterWidth/2 + blockDim.x)] = inputChannel[(Y + blockDim.y) * numCols + X + blockDim.x];
	}

	if(threadIdx.y < filterWidth/2)
	{
		data[threadIdx.y * width + (threadIdx.x + filterWidth/2)] = inputChannel[(Y - filterWidth/2) * numCols + X];
		data[(threadIdx.y + 2 * blockDim.y + filterWidth/2) * width + (threadIdx.x + filterWidth/2)] = inputChannel[(Y + 2 * blockDim.y) * numCols + X];
	}

	if(threadIdx.x < filterWidth/2 && threadIdx.y < filterWidth/2)
	{
		data[threadIdx.y * width + threadIdx.x] = inputChannel[(Y - filterWidth/2) * numCols + X - filterWidth/2];
		data[threadIdx.y * width + (threadIdx.x + filterWidth/2 + blockDim.x)] = inputChannel[(Y + blockDim.y) * numCols + X + blockDim.x];
		data[(threadIdx.y + 2 * blockDim.y + filterWidth/2) * width + threadIdx.x] = inputChannel[(Y + 2 * blockDim.y) * numCols + X - filterWidth/2];
		data[(threadIdx.y + 2 * blockDim.y + filterWidth/2) * width + (threadIdx.x + filterWidth/2 + blockDim.x)] = inputChannel[(Y + 2 * blockDim.y) * numCols + X + blockDim.x];
	}
	__syncthreads();

	//For every value in the filter around the pixel (c, r)
	float result[] = {0, 0};
	for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r)
	{
		for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c)
		{
			//Find the global image position for this filter position
			//clamp to boundary of the image
			int image_r = threadIdx.y + filterWidth/2 + filter_r;
			int image_c = threadIdx.x + filterWidth/2 + filter_c;
			float image_value = data[image_r * width + image_c];

			int image_r1 = threadIdx.y + blockDim.y + filterWidth/2 + filter_r;
			float image_value1 = data[image_r1 * width + image_c];

			float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + (filter_c + filterWidth/2)];
			result[0] += image_value * filter_value;
			result[1] += image_value1 * filter_value;
		}
	}
	outputChannel[Y * numCols + X] = result[0];
	outputChannel[(Y + blockDim.y) * numCols + X] = result[1];
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
	__global__
void separateChannels(const uchar4* const inputImageRGBA,
		int numRows,
		int numCols,
		unsigned char* const redChannel,
		unsigned char* const greenChannel,
		unsigned char* const blueChannel)
{
	//
	// NOTE: Be careful not to try to access memory that is outside the bounds of
	// the image. You'll want code that performs the following check before accessing
	// GPU memory:
	//
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( X >= numCols || Y >= numRows ) { return; }
	int idx = Y * numCols + X;
	uchar4 pixel = inputImageRGBA[idx];
	redChannel[idx] = pixel.x;
	greenChannel[idx] = pixel.y;
	blueChannel[idx] = pixel.z;

}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
	__global__
void recombineChannels(const unsigned char* const redChannel,
		const unsigned char* const greenChannel,
		const unsigned char* const blueChannel,
		uchar4* const outputImageRGBA,
		int numRows,
		int numCols)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
			blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red   = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue  = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
		const float* const h_filter, const size_t filterWidth)
{

	//allocate memory for the three different channels
	//original
	//checkCudaErrors(cudaMallocManaged(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
	//checkCudaErrors(cudaMallocManaged(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	//checkCudaErrors(cudaMallocManaged(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

	//Allocate memory for the filter on the GPU
	//Use the pointer d_filter that we have already declared for you
	//You need to allocate memory for the filter with cudaMalloc
	//be sure to use checkCudaErrors like the above examples to
	//be able to tell if anything goes wrong
	//IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
	checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));

	//Copy the filter on the host (h_filter) to the memory you just allocated
	//on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
	//Remember to use checkCudaErrors!
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(uchar4 * const d_inputImageRGBA,
		uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
		unsigned char *d_redBlurred, 
		unsigned char *d_greenBlurred, 
		unsigned char *d_blueBlurred,
		const int filterWidth)
{
	// Set reasonable block size (i.e., number of threads per block)
	const dim3 blocksize(32, 16, 1);

	//Compute correct grid size (i.e., number of blocks per kernel launch)
	//from the image size and and block size.
	const dim3 gridsize(20, 18, 1);

	// Launch a kernel for separating the RGBA image into different color channels
	separateChannels<<<gridsize, blocksize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

	// Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
	// launching your kernel to make sure that you didn't make any mistakes.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Call your convolution kernel here 3 times, once for each color channel.
	gaussian_blur<<<gridsize, blocksize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
	gaussian_blur<<<gridsize, blocksize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	gaussian_blur<<<gridsize, blocksize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

	// Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
	// launching your kernel to make sure that you didn't make any mistakes.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Now we recombine your results. We take care of launching this kernel for you.
	//
	// NOTE: This kernel launch depends on the gridSize and blockSize variables,
	// which you must set yourself.
	recombineChannels<<<gridsize, blocksize>>>(d_redBlurred,
			d_greenBlurred,
			d_blueBlurred,
			d_outputImageRGBA,
			numRows,
			numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_filter));
}