#include "util.h"
#include <iostream>
using namespace std;

__host__ void cuda_check(string file, int line)
{
	static string prev_file = "";
	static int prev_line = 0;
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}

__host__ void init_device()
{
	// Before the GPU can process your kernels, a so called "CUDA context" must be initialized
	// This happens on the very first call to a CUDA function, and takes some time (around half a second)
	// We will do it right here, so that the run time measurements are accurate

	cudaDeviceSynchronize();
	CUDA_CHECK;
}


__device__ __host__ int clamp(int i, int lo, int hi)
{
	return max(lo, min(i, hi));
}

__device__ __host__ int clamp_address(int x, int y, int w, int h)
{
	int xaddr = clamp(x, 0, w - 1);
	int yaddr = clamp(y, 0, h - 1);
	return xaddr + yaddr * w;
}

__device__ __host__ int clamp_address(int x, int y, int c, int w, int h, int nc)
{
	int xaddr = clamp(x, 0, w - 1);
    int yaddr = clamp(y, 0, h - 1);
    int caddr = clamp(c, 0, nc - 1);
	return xaddr + yaddr * w + caddr * w * h;
}

template <class T>
__device__ T read_data(const T *I, int w, int h, int nc, int atx, int aty, int atc)
{
	return I[clamp_address(atx, aty, atc, w, h, nc)];
}

template <class T>
__device__ T read_data(const T *I, int w, int h, int atx, int aty)
{
	return I[clamp_address(atx, aty, w, h)];
}

template <class T>
__device__ void write_data(T *I, const T data, int w, int h, int nc, int atx, int aty, int atc)
{
    I[clamp_address(atx, aty, atc, w, h, nc)] = data;
}

template <class T>
__device__ void write_data(T *I, const T data, int w, int h, int atx, int aty)
{
	I[clamp_address(atx, aty, w, h)] = data;
}

__device__ float read_data(const float *I, int w, int h, int nc, int atx, int aty, int atc)
{
	return read_data<float>(I, w, h, nc, atx, aty, atc);
}

__device__ float read_data(const float *I, int w, int h, int atx, int aty)
{
	return read_data<float>(I, w, h, atx, aty);
}

__device__ int read_data(const int *I, int w, int h, int nc, int atx, int aty, int atc)
{
	return read_data<int>(I, w, h, nc, atx, aty, atc);
}

__device__ int read_data(const int *I, int w, int h, int atx, int aty)
{
	return read_data<int>(I, w, h, atx, aty);
}

__device__ void write_data(float *I, const float data, int w, int h, int nc, int atx, int aty, int atc)
{
	write_data<float>(I, data, w, h, nc, atx, aty, atc);
}

__device__ void write_data(float *I, const float data, int w, int h, int atx, int aty)
{
	write_data<float>(I, data, w, h, atx, aty);
}

__device__ void write_data(int *I, const int data, int w, int h, int nc, int atx, int aty, int atc)
{
	write_data<int>(I, data, w, h, nc, atx, aty, atc);
}

__device__ void write_data(int *I, const int data, int w, int h, int atx, int aty)
{
	write_data<int>(I, data, w, h, atx, aty);
}

__device__ __host__ float square(float a)
{
	return a * a;
}


