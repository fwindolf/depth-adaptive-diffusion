#ifndef UTIL_H
#define UTIL_H

#include <cuda_runtime.h>
#include <string>

/**
 * Makro for cuda error checking
 */
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)

/**
 * Check if last cuda call produced an error and print it
 */
__host__ void cuda_check(std::string file, int line);

/**
 * Initialize device
 */
__host__ void init_device();


/*
* Clamp a value between lo and hi
*/
__device__ __host__ int clamp(int i, int lo, int hi);

/**
 * Clamp a float between lo and hi
 */
__device__ __host__ float fclamp(float i, float lo, float hi);

/*
* Clamp an address for x and y
*/
__device__ __host__ int clamp_address(int x, int y, int w, int h);

/*
* Clamp an address for x, y and c
*/
__device__ __host__ int clamp_address(int x, int y, int c, int w, int h, int nc);

/**
* Read data from a float image (x,y,c) with clamped coordinates
*/
__device__ float read_data(const float *I, int w, int h, int nc, int atx, int aty, int atc);

/**
* Read data from an integer value image (x,y,c) with clamped coordinates
*/
__device__ int read_data(const int *I, int w, int h, int nc, int atx, int aty, int atc);

/**
* Read data from a float image (x,y) with clamped coordinates
*/
__device__ float read_data(const float *I, int w, int h, int atx, int aty);

/**
* Read data from an integer value image (x,y) with clamped coordinates
*/
__device__ int read_data(const int *I, int w, int h, int atx, int aty);

/**
* Write data to a float image (x,y,c) with clamped coordinates
*/
__device__ void write_data(float *I, const float data, int w, int h, int nc, int atx, int aty, int atc);

/**
* Write data to an integer image (x,y,c) with clamped coordinates
*/
__device__ void write_data(int *I, const int data, int w, int h, int nc, int atx, int aty, int atc);

/**
* Write data to a float image (x,y) with clamped coordinates
*/
__device__ void write_data(float *I, const float data, int w, int h, int atx, int aty);

/**
* Write data to a float image (x,y) with clamped coordinates
*/
__device__ void write_data(int *I, const int data, int w, int h, int atx, int aty);


/**
 * Square a number
 */
__device__ __host__ float square(float a);

#endif // UTIL_H
