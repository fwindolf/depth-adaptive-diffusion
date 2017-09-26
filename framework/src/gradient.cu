#include "gradient.h"
#include "util.h"

__global__ void g_l2norm(float * I, float *O, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.f;
    if(nc > 1)
    {
        for(int c = 0; c < nc; c++)
        {
            sum += square(read_data(I, w, h, nc, x, y, c));
        }
        write_data(O, sqrtf(sum), w, h, x, y);
    }
    else
    {
    	write_data(O, read_data(I, w, h, x, y), w, h, x, y);
    }
}

__global__ void g_l2norm(float * V_1, float * V_2, float *O, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.f;

    for(int c = 0; c < nc; c++)
    {
    	sum += square(read_data(V_1, w, h, nc, x, y, c));
		sum += square(read_data(V_2, w, h, nc, x, y, c));
    }
    write_data(O, sqrtf(sum), w, h, x, y);
}

__global__ void g_gradient(float *I, float *V_1, float *V_2, int w, int h, int nc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z + blockDim.z * blockIdx.z;

	float diff;
	float field;

	if(x < w - 1 && y < h - 1 && c < nc)
	{
		field = read_data(I, w, h, nc, x, y, c);
		// calculate the gradient with forward difference
		diff = read_data(I, w, h, nc, x + 1, y, c) - field;
		write_data(V_1, diff, w, h, nc, x, y, c);
		diff = read_data(I, w, h, nc, x, y + 1, c) - field;
		write_data(V_2, diff, w, h, nc, x, y, c);
	}
}


__global__ void g_divergence(float * V_1, float *V_2, float *D, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h )
    {
        // Compute the divergence as the sum of backwards differences from the gradient parts
		// backward difference in x direction of first gradient part
        float div_1 = read_data(V_1, w, h, nc, x, y, c) - read_data(V_1, w, h, nc, x - 1, y, c);

        // backward difference in y direction of second gradient part
        float div_2 = read_data(V_2, w, h, nc, x, y, c) - read_data(V_2, w, h, nc, x, y - 1, c);

        write_data(D, div_1 + div_2, w, h, nc, x, y, c);
    }
}

