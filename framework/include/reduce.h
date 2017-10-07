#ifndef REDUCE_H
#define REDUCE_H

#include "cuda_runtime.h"


// Define the function structure for reduces

/**
 * Normalize the data in Data to interval [o_min, o_max] for one channel
 */
__host__ void normalize(float *Data, int w, int h, float o_min, float o_max);


#endif // REDUCE_H
