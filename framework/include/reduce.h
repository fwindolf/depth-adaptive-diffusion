#ifndef REDUCE_H
#define REDUCE_H

#include "cuda_runtime.h"
#include <iostream>


// Define the function structure for reduces
//typedef float (*reduce_fn)(float, float);


/**
 * Normalize the data in Data to interval [o_min, o_max] for one channel
 */
__host__ void normalize(float *Data, int w, int h, float o_min, float o_max);


/**
 * Kernel for normalizing the Data array with linear transform
 * d = a * d + b
 */
__global__ void g_normalize(float * Data, int w, int h, float a, float b);


#endif // REDUCE_H

