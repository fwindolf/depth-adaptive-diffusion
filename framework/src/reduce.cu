#include "util.h"
#include "reduce.h"
#include <iostream>
#include <algorithm>

using namespace std;

__global__ void g_normalize(float * Data, int w, int h, float a, float b)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		// Normalize via linear transform
		float d = read_data(Data, w, h, x, y);
		write_data(Data, a * d + b, w, h, x, y);
	}
}

__host__ void normalize(float *Data, int w, int h, float o_min, float o_max)
{
	// Create data array on host for sorting with stl
	size_t d_dim = w * h;
	float d_host[d_dim];

	cudaMemcpy(&d_host, Data, d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);

	// Sort array in ascending order
	sort(d_host, d_host + d_dim);

	double r_min = d_host[0];
	double r_max = d_host[d_dim - 1];

	// Normalize the data in kernel
	dim3 block2D(128, 2);
	dim3 grid2D((w + block2D.x - 1) / block2D.x,
			(h + block2D.y - 1) / block2D.y);

	// Calculate coefficients
	float a = (o_max - o_min) / (r_max - r_min);
	float b = o_max - a * r_max;

	g_normalize<<<grid2D, block2D>>>(Data, w, h, a, b);

}

