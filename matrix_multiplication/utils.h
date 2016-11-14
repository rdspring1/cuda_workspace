#ifndef UTILS_H__
#define UTILS_H__

#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

template<typename T>
void checkResultsExact(const T* const left, const T* const right, size_t numElem)
{
	for (size_t i = 0; i < numElem; ++i) 
	{
		if (left[i] != right[i]) 
		{
			std::cerr << "Difference at pos " << i << std::endl;
			std::cerr << "left: \t" << std::setprecision(17) << +left[i] << "\nright: \t" << +right[i] << std::endl;
			exit(1);
		}
	}
}
#endif
