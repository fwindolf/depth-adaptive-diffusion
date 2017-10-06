#include "util.h"
#include "reduce.h"
#include <iostream>
/*
__device__ float d_sum(float cur, float add)
{
	return cur + add;
}

__device__ float d_max(float cur, float val)
{
	return max(cur, val);
}

__device__ float d_min(float cur, float val)
{
	return min(cur, val);
}

__device__ reduce_fn p_reduce_sum = d_sum;
__device__ reduce_fn p_reduce_min = d_min;
__device__ reduce_fn p_reduce_max = d_max;

__global__ void g_reduce(float * data, int offset, int items, reduce_fn f)
{
	extern __shared__ float sm[];

	// calculate global and index in block
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int t = threadIdx.x;

	if (x * offset < items)
	{
		// get data into shared memory
		sm[t] = data[x];
	}
	else
	{
		sm[t] = 0;
	}

	// delay until every thread has finished reading global memory
	__syncthreads();

	// reduce whole block while taking care of bank conflicts
	// always
	for (int s = blockDim.x / 2; s > 0; s /= 2)
	{
		// increasing amount of threads goes idle
		if (t < s)
		{
			// apply function
			sm[t] = f(sm[t], sm[t + s]);
		}
		__syncthreads();
	}

	// write result left aligned to global memory
	if (t == 0)
	{
		data[blockIdx.x] = sm[t];
	}
}
*/
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
/*
__host__ void normalize(float *Data, int w, int h, float o_min, float o_max)
{
	int items = w * h;

	// Create as many blocks as needed
	dim3 block(256, 1, 1);
	int nblocks = ceil((float) items / block.x);
	dim3 grid(nblocks, 1, 1);

	// As least as much space as all blocks will read valid data (0)
	size_t nbytes = nblocks * block.x * sizeof(float);
	// Shared memory for one block
	size_t smbytes = block.x * sizeof(float);

	// Reserve space on device
	float * I = NULL;

	cudaMalloc(&I, nbytes);
	CUDA_CHECK;

	cudaMemset(I, 0.f, nbytes);
	CUDA_CHECK;

	// Make a copy of the data
	cudaMemcpy(I, Data, w * h * sizeof(float), cudaMemcpyDeviceToDevice);
	CUDA_CHECK;

	// Create the static function pointers on device
	reduce_fn max_fn;
	reduce_fn min_fn;
	cudaMemcpyFromSymbol( &max_fn, p_reduce_max, sizeof( reduce_fn ) );
	cudaMemcpyFromSymbol( &min_fn, p_reduce_min, sizeof( reduce_fn ) );

	int offset = 1;
	// Per iteration we can reduce 32 * blockdimension elements
	for (int i = items; i > 1; i = ceil(float(i) / block.x))
	{
		// In the first level the offset is 0, then the first element is
		// every block.x entries in the global data, and so on

		nblocks = ceil((float) i / block.x);
		dim3 grid(nblocks, 1, 1);
		g_reduce<<<grid, block, smbytes>>>(I, offset, items, max_fn);
		CUDA_CHECK;

		// Increase offset
		offset *= block.x;
		cudaDeviceSynchronize();
	}

	float r_max = 0.f;
	// copy data to device
	cudaMemcpy(&r_max, &I[0], sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<r_max<<"\n";
	CUDA_CHECK;

	cudaMemset(I, 0.f, nbytes);
	CUDA_CHECK;

	// Make a copy of the data
	cudaMemcpy(I, Data, w * h * sizeof(float), cudaMemcpyDeviceToDevice);
	CUDA_CHECK;

	offset = 1;
	// Per iteration we can reduce 32 * blockdimension elements
	for (int i = items; i > 1; i = ceil(float(i) / block.x))
	{
		// In the first level the offset is 0, then the first element is
		// every block.x entries in the global data, and so on

		nblocks = ceil((float) i / block.x);
		dim3 grid(nblocks, 1, 1);
		g_reduce<<<grid, block, smbytes>>>(I, offset, items, min_fn);
		CUDA_CHECK;

		// Increase offset
		offset *= block.x;
		cudaDeviceSynchronize();
	}

	float r_min = 0.f;
	// copy data to device
	cudaMemcpy(&r_min, &I[0], sizeof(float), cudaMemcpyDeviceToHost);
    
	CUDA_CHECK;

	// Normalize the data
	dim3 block2D(128, 2);
	dim3 grid2D((w + block2D.x - 1) / block2D.x,
			(h + block2D.y - 1) / block2D.y);

	// Calculate coefficients
	float a = (o_max - o_min)/(r_max - r_min);
	float b = o_max - a * r_max;
	g_normalize<<<grid2D, block2D>>>(Data, w, h, a, b);
}
*/
__host__ void normalize(float *Data, int w, int h, float o_min, float o_max)

{
  float *dept = new float[(size_t) w * h];
  cudaMemcpy(dept,Data,(size_t)(w * h) * sizeof(float), cudaMemcpyDeviceToHost);
  float tmp =0.f;
  // Sorting an array
  for(int i=0;i<(w*h);i++)
    {
     for(int j=i+1;j<(w*h);j++)
       {
        if(dept[i]>dept[j])
            {
              tmp=dept[i];
              dept[i]=dept[j];
              dept[j]=tmp;            
            }
      }


   }
  //printing sorted values
  /*for(int i=0;i<(w*h);i++)

     {

      std::cout<<dept[i]<<"\t";

      }  
   */
  double r_min=dept[0];
  double r_max=dept[(w*h)-1];
  //std::cout<<"minimum is"<<r_min<<"\n";
  //std::cout<<"maximum is"<<r_max<<"\n";
   
  // Normalize the data
  dim3 block2D(128, 2);
  dim3 grid2D((w + block2D.x - 1) / block2D.x, (h + block2D.y - 1) / block2D.y);

  // Calculate coefficients
  float a = (o_max - o_min)/(r_max - r_min);
  float b =  o_max - a * r_max;
  //std::cout<<"\n"<<a<<"\t"<<b<<"\t";
  g_normalize<<<grid2D, block2D>>>(Data, w, h, a, b);







}



