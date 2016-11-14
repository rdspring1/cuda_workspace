all:
	nvcc -o test test.cu -lcublas -lcurand -O3 -std=c++11 -arch=sm_52 -Xcompiler -Wall -Xcompiler -Wextra -m64 --default-stream per-thread
