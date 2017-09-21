#include "util.h"
#include "usage.h"
#include "image.h"
#include <iostream>
#include <iomanip>
using namespace std;

// TODO: gamma_min - gamma_max 0 indexing (g_compute_g, ...?)

/**
 * Rho function implementation
 *
 * Calculates the error in at the pixel
 * iL is the pixel values at x, y of the left image for all channels nc
 * iR is the pixel values at x + gamma (current disparity) of the right image for all nc
 * lambda is some constant parameter
 */
__device__ float rho(float *iL, float *iR, int nc, float lambda)
{
	float sum = 0.f;
	// Sum the error for all channels
	for (int c = 0; c < nc; c++)
	{
		sum += iL[c] - iR[c];
	}
	return sum * lambda;
}


/**
 * Initialize P to value
 *
 * x * y * gc threads needed
 */
__global__ void g_initialize_p(float * P, int w, int h, int gc, float value)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		write_data(P, value, w, h, 3 * gc, x, y, 1*g);
		write_data(P, value, w, h, 3 * gc, x, y, 2*g);
		write_data(P, value, w, h, 3 * gc, x, y, 3*g);
	}
}


/**
 * Initialize Phi to value
 *
 * x * y * gc threads needed
 */
__global__ void g_initialize_phi(float * Phi, int w, int h, int gc, float value)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		write_data(Phi, value, w, h, gc, x, y, g);
	}
}


/**
 * Project the P vector back onto C
 *
 * x * y * gc threads needed
 *
 * P = (p1, p2, p3) with p1 = (w*h*gc)
 * IL and IR are the left/right original images with x * y * nc
 * G contains the disparity for each pixel [gamma_min ... gamma_max] = gc
 */
__global__ void g_project_p_c(float * P, float * IL, float *IR, int w,
		int h, int nc, int gc, float lambda, float gamma_min)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		// p1..2 must hold true to the constraint that sqrt(p1² + p2²) <= 1
		// p3 must hold true to the constraint that |p3| <= rho(x, gamma)
		int max_z = 3 * gc;
		int idx_p1_z = 0 * g;
		int idx_p2_z = 1 * g;
		int idx_p3_z = 2 * g;

		float p1 = read_data(P, w, h, max_z, x, y, idx_p1_z);
		float p2 = read_data(P, w, h, max_z, x, y, idx_p2_z);
		float p3 = read_data(P, w, h, max_z, x, y, idx_p3_z);

		// p1, p2
		p1 = p1 / max(1.f, sqrtf(square(p1) + square(p2)));
		p2 = p2 / max(1.f, sqrtf(square(p1) + square(p2)));

		// p3
		float iL[3];
		float iR[3];

		// Save image data to temporary arrays
		for (int c = 0; c < nc; c++)
		{
			iL[c] = read_data(IL, w, h, nc, x, y, c);
			// Use the disparity value of this layer of P
			iR[c] = read_data(IR, w, h, nc, x + gamma_min + g, y, c);
		}

		// p3
		p3 = p3 / max(1.f, fabs(p3)/rho(iL, iR, nc, lambda));

		// write the results back to P
		write_data(P, p1, w, h, max_z, x, y, idx_p1_z);
		write_data(P, p2, w, h, max_z, x, y, idx_p2_z);
		write_data(P, p3, w, h, max_z, x, y, idx_p3_z);
	}
}

/**
 * Project the Phi vector back onto D
 *
 * x * y * gc threads needed
 *
 * Phi is dimensions (w * h * gc) and can be in [0, 1], thus the value gets
 * clamped back into the interval in the projection
 */
__global__ void g_project_phi_d(float * Phi, int w, int h, int gc)
{
	// phi must be truncated to the interval [0,1]
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		float phi = read_data(Phi, w, h, gc, x, y, g);
		// TODO: Is clamping to [0, 1] enough or is there something to do with gamma_min => 0, gamma_max => 1?
		write_data(Phi, clamp(phi, 0, 1), w, h, gc, x, y, g);
	}
}

/**
 * Update the P vector and save the result back to P
 *
 * x * y * gc threads needed
 *
 * P = (p1, p2, p3) with dimensions (w * h * gc * 3)
 * Grad3_Phi is the gradient of Phi in x, y and gamma direction (w * h * gc * 3)
 */
__global__ void g_update_p(float * P, float *Grad3_Phi, int w, int h, int gc,
		float tau_d)
{
	// p^k+1 = PC(p^k + tau_d * grad3(Phi))
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		// p has 3 channels
		int pc = 3;
		// maximum z index for P and Grad3_Phi
		int max_z = 3 * gc;

		float p_next;
		for (int i = 0; i < pc; i++)
		{
			int idx_z = i * gc;
			p_next = read_data(P, w, h, max_z, x, y, idx_z)
					+ tau_d * read_data(Grad3_Phi, w, h, max_z, x, y, idx_z);

			// Write back to P
			write_data(P, p_next, w, h, max_z, x, y, idx_z);
		}
	}
}

/**
 * Update the Phi vector and save the result back to Phi
 *
 * x * y * gc threads needed
 *
 * Phi and Div3_P are dimensions (w * h * gc)
 */
__global__ void g_update_phi(float *Phi, float *Div3_P, int w, int h, int gc,
		float tau_p)
{
	// phi^k+1 = PD(phi^k + div3(p^k))
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		float phi_next = read_data(Phi, w, h, gc, x, y, g)
				+ tau_p * read_data(Div3_P, w, h, gc, x, y, g);

		// Write back to Phi
		write_data(Phi, phi_next, w, h, gc, x, y, g);
	}
}

/**
 * Calculate the gradient in x, y and gamma direction
 * Phi is dimensions (w * h * gc)
 *
 * x * y * gc threads needed
 *
 * Grad3_Phi is the resulting w * h * gc * 3 with one channel for x, y and g direction
 */
__global__ void g_grad3(float *Phi, float *Grad3_Phi, int w, int h, int gc)
{
	// Gradient 3 is defined via forward differences
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		float phi = read_data(Phi, w, h, gc, x, y, g);
		float dx = read_data(Phi, w, h, gc, x + 1, y, g) - phi;
		float dy = read_data(Phi, w, h, gc, x, y + 1, g) - phi;
		float dg = read_data(Phi, w, h, gc, x, y, g + 1) - phi;

		// 3 channels on the gradient
		int max_z = 3 * gc;
		int idx_phi_x = 0 * g;
		int idx_phi_y = 1 * g;
		int idx_phi_g = 2 * g;

		// Write the forward differences in different directions stacked into phi
		write_data(Grad3_Phi, dx, w, h, max_z, x, y, idx_phi_x);
		write_data(Grad3_Phi, dy, w, h, max_z, x, y, idx_phi_y);
		write_data(Grad3_Phi, dg, w, h, max_z, x, y, idx_phi_g);
	}
}

/**
 * Calculate the divergence of P
 *
 * x * y * gc threads needed
 *
 * P is stored in (w * h * gc * 3), so (p1, p2, p3)
 * Div3_P has the same dimensions as Phi (w * h * gc)
 */
__global__ void g_div3(float *P, float *Div3_P, int w, int h, int gc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int g = threadIdx.z + blockDim.z * blockIdx.z;

	if (x < w && y < h && g < gc)
	{
		// Calculate the indices for p1, p2, p3
		int max_z = 3 * gc;
		int idx_p1_z = 0 * g;
		int idx_p2_z = 1 * g;
		int idx_p3_z = 2 * g;
		// create last index, that may only lie in the range of the p3 index, thus clamp manually
		int idx_p3_z_1 = clamp(idx_p3_z - 1, 2 * gc, (3 * gc) - 1);

		float p1 = read_data(P, w, h, max_z, x, y, idx_p1_z);
		float p2 = read_data(P, w, h, max_z, x, y, idx_p2_z);
		float p3 = read_data(P, w, h, max_z, x, y, idx_p3_z);

		// Divergence 3 is defined as the sum of backward differences
		float div = p1 - read_data(P, w, h, max_z, x - 1, y, idx_p1_z) + p2
				- read_data(P, w, h, max_z, x, y - 1, idx_p2_z) + p3
				- read_data(P, w, h, max_z, x, y, idx_p3_z_1);

		write_data(Div3_P, div, w, h, x, y);
	}
}

/**
 * Calculate the function u(x) (=G) from Phi
 */
__global__ void g_compute_g(float *Phi, float *G, int w, int h, int gamma_min,
		int gamma_max)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	float gamma = gamma_min;
	int gc = gamma_max - gamma_min;

	for (int g = 0; g < gc; g++)
	{
		// TODO: maybe Phi(x, gamma) is {0, 1} not [0, 1] -> use mu to round
		gamma += read_data(Phi, w, h, gc, x, y, g);
	}

	write_data(G, gamma, w, h, x, y);
}

int main(int argc, char **argv)
{
	init_device();

	// Define parameters
	string image;
	bool gray;
	float lambda, tau_p, tau_d;
	int gamma_min, gamma_max, max_iterations;

	read_parameters(image, gray, lambda, tau_p, tau_d, gamma_min, gamma_max,
			max_iterations, argc, argv);

	// define the range of gamma
	int gc = gamma_max - gamma_min;

	// image + 0 is left
	string imageL = image + "0.png";
	// image + 1 is right
	string imageR = image + "1.png";

	cv::Mat mInL = load_image(imageL, gray);
	cv::Mat mInR = load_image(imageR, gray);

	// Width, height and channels of image
	int w, h, nc;
	get_dimensions(mInL, mInR, w, h, nc);
	cout << "Image Dimensions: " << w << " x " << h << " x " << nc << endl;

	// Define output array, grayscale image of depth values
	cv::Mat mOut(h, w, CV_32FC1);

	// allocate raw input image array
	float *imgInL = new float[(size_t) w * h * nc];
	float *imgInR = new float[(size_t) w * h * nc];

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t) w * h * mOut.channels()];

	// Init raw input image array
	convert_mat_to_layered(imgInL, mInL);
	convert_mat_to_layered(imgInR, mInR);

	// Allocate memory on device for images
	size_t imgBytes = w * h * nc * sizeof(float);
	float *IL, *IR = NULL;
	cudaMalloc(&IL, imgBytes);
	CUDA_CHECK;
	cudaMalloc(&IR, imgBytes);
	CUDA_CHECK;
	cudaMemset(IL, 0, imgBytes);
	CUDA_CHECK;
	cudaMemset(IR, 0, imgBytes);
	CUDA_CHECK;

	// for P (3 channels for p1..3) and Phi vectors
	size_t pBytes = w * h * gc * 3 * sizeof(float);
	size_t phiBytes = w * h * gc * sizeof(float);
	float *P, *Phi = NULL;
	cudaMalloc(&P, pBytes);
	CUDA_CHECK;
	cudaMalloc(&Phi, phiBytes);
	CUDA_CHECK;
	cudaMemset(P, 0, pBytes);
	CUDA_CHECK;
	cudaMemset(Phi, 0, phiBytes);
	CUDA_CHECK;

	// for grad3 of phi and div3 of p
	float * Grad3_Phi, *Div3_P = NULL;
	cudaMalloc(&Grad3_Phi, pBytes);
	CUDA_CHECK;
	cudaMalloc(&Div3_P, phiBytes);
	CUDA_CHECK;
	cudaMemset(Grad3_Phi, 0, pBytes);
	CUDA_CHECK;
	cudaMemset(Div3_P, 0, phiBytes);
	CUDA_CHECK;

	// for the final depth values
	size_t gBytes = w * h * sizeof(float);
	float * G = NULL;
	cudaMalloc(&G, gBytes);
	CUDA_CHECK;
	cudaMemset(G, 0, gBytes);

	// copy data to device
	cudaMemcpy(IL, imgInL, imgBytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(IR, imgInR, imgBytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;

	int blockx = 128;
	int blocky = 2;

	// create kernel dimensions
	dim3 block2D(blockx, blocky);
	dim3 grid2D((w + block2D.x - 1) / block2D.x,
			(h + block2D.y - 1) / block2D.y);

	dim3 block3D(blockx, blocky, 1);
	dim3 grid3D((w + block3D.x - 1) / block3D.x,
			(h + block3D.y - 1) / block3D.y, (gc + block3D.z - 1) / block3D.z);

	// Actual Algorithm
	// Initialization of P and Phi
	g_initialize_phi<<<grid3D, block3D>>>(Phi, w, h, gc, 0.5f);
	CUDA_CHECK;

	g_initialize_p<<<grid3D, block3D>>>(P, w, h, gc, 0.5f);
	CUDA_CHECK;

	// Make sure the initial Phi fits constraints
	g_project_phi_d<<<grid3D, block3D>>>(Phi, w, h, gc);
	CUDA_CHECK;

	// Make sure the initial P fits constraints
	g_project_p_c<<<grid3D, block3D>>>(P, IL, IR, w, h, nc, gc, lambda, gamma_min);
	CUDA_CHECK;

	// Iterate until stopping criterion is reached
	int iterations = 0;
	while (1)
	{
		// Calculate the divergence of P for the update step of phi
		g_div3<<<grid3D, block3D>>>(P, Div3_P, w, h, gc);
		CUDA_CHECK;

		// Update the Phi
		g_update_phi<<<grid3D, block3D>>>(Phi, Div3_P, w, h, gc, tau_p);
		CUDA_CHECK;

		// Make sure Phi is in the solution space (D)
		g_project_phi_d<<<grid3D, block3D>>>(Phi, w, h, gc);
		CUDA_CHECK;

		// Calculate the gradient in x, y, and gamma direction
		g_grad3<<<grid3D, block3D>>>(Phi, Grad3_Phi, w, h, gc);
		CUDA_CHECK;

		// Update the P
		g_update_p<<<grid3D, block3D>>>(P, Grad3_Phi, w, h, gc, tau_d);
		CUDA_CHECK;

		// Make sure P is in solution space (C)
		g_project_p_c<<<grid3D, block3D>>>(P, IL, IR, w, h, nc, gc, lambda, gamma_min);
		CUDA_CHECK;

		if (iterations > max_iterations)
			break;

		iterations++;
	}

	// Calculate the new G
		g_compute_g<<<grid2D, block2D>>>(Phi, G, w, h, gamma_min, gamma_max);


	// Move disparities from device to host
	cudaMemcpy(imgOut, Phi, gBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// show input image
	showImage("Input", mInL, 100, 100); // show at position (x_from_left=100,y_from_above=100)

	// show output image: first convert to interleaved opencv format from the layered raw array
	convert_layered_to_mat(mOut, imgOut);
	// cv::normalize(mOut, mOut, 0, 1);
	showImage("Output", mOut, 100 + w + 40, 100);

	// free allocated arrays
	delete[] imgInL;
	delete[] imgInR;
	delete[] imgOut;

	// wait for key inputs
	cv::waitKey(0);

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}
