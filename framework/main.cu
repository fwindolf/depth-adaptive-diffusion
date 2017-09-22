#include "util.h"
#include "usage.h"
#include "image.h"
#include <iostream>
#include <vector>
using namespace std;

/**
 * Rho function implementation
 *
 * Calculates the absolute error for each pixel
 *
 * iL is the pixel values at x, y of the left image for all channels nc
 * iR is the pixel values at x  plus a current disparity of the right image for all nc
 * lambda is some constant parameter
 */
__device__ float rho(float *iL, float *iR, int nc, float lambda)
{
	float sum = 0.f;
	// Sum the error for all channels
	for (int c = 0; c < nc; c++)
	{
		sum += fabs(iL[c] - iR[c]);
	}
	return sum * lambda;
}

__global__ void g_test_rho(float * IL, float * IR, float *IO, int w, int h,
		int nc, int gc, int gamma_min, float lambda)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if(x < w && y < h)
	{
		for (int g = 0; g < gc; g++)
		{
			float iL[3];
			float iR[3];
			// Save image data to temporary arrays
			for (int c = 0; c < nc; c++)
			{
				iL[c] = read_data(IL, w, h, nc, x, y, c);
				// Use the disparity value of this layer of P
				// index of gamma runs from 0...gc, thus offset by gamma_min (eg. -16)
				iR[c] = read_data(IR, w, h, nc, x + gamma_min + g, y, c);
			}
			write_data(IO, rho(iL, iR, nc, lambda), w, h, gc, x, y, g);
		}
	}
}

/**
 * Initialize Phi to value
 *
 * x threads needed
 */
__global__ void g_initialize_phi(float * Phi, int w, int h, int gc, float value)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x < w * h * gc)
	{
		Phi[x] = value;
	}
}

/**
 * Project the P vector back onto C
 *
 * x * y threads needed
 *
 * P = (p1, p2, p3) with p1 = (w*h*gc)
 * IL and IR are the left/right original images with x * y * nc
 * G contains the disparity for each pixel [gamma_min ... gamma_max] = gc
 */
__global__ void g_project_p_c(float * P, float * IL, float *IR, int w, int h,
		int nc, int gc, float lambda, float gamma_min)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		// Consider every disparity channel
		for (int g = 0; g < gc; g++)
		{
			// p1..2 must hold true to the constraint that sqrt(p1² + p2²) <= 1
			// p3 must hold true to the constraint that |p3| <= rho(x, gamma)
			int max_z = 3 * gc;
			int idx_p1_z = 0 * gc + g;
			int idx_p2_z = 1 * gc + g;
			int idx_p3_z = 2 * gc + g;

			float p1 = read_data(P, w, h, max_z, x, y, idx_p1_z);
			float p2 = read_data(P, w, h, max_z, x, y, idx_p2_z);
			float p3 = read_data(P, w, h, max_z, x, y, idx_p3_z);

			// p1, p2
			float tmp = max(1.f, sqrtf(square(p1) + square(p2)));
			p1 = p1 / tmp;
			p2 = p2 / tmp;

			// p3
			float iL[3];
			float iR[3];

			// Save image data to temporary arrays
			for (int c = 0; c < nc; c++)
			{
				iL[c] = read_data(IL, w, h, nc, x, y, c);
				// Use the disparity value of this layer of P
				// index of gamma runs from 0...gc, thus offset by gamma_min (eg. -16)
				iR[c] = read_data(IR, w, h, nc, x + gamma_min + g, y, c);
			}

			// p3
			p3 = p3 / max(1.f, fabs(p3) / rho(iL, iR, nc, lambda));

			// write the results back to P
			write_data(P, p1, w, h, max_z, x, y, idx_p1_z);
			write_data(P, p2, w, h, max_z, x, y, idx_p2_z);
			write_data(P, p3, w, h, max_z, x, y, idx_p3_z);
		}
	}
}

/**
 * Project the Phi vector back onto D
 *
 * x * y threads needed
 *
 * Phi is dimensions (w * h * gc) and can be in [0, 1], thus the value gets
 * clamped back into the interval in the projection
 */
__global__ void g_project_phi_d(float * Phi, int w, int h, int gc)
{
	// phi must be truncated to the interval [0,1]
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		for (int g = 1; g < gc - 1; g++)
		{
			float phi = read_data(Phi, w, h, gc, x, y, g);
			write_data(Phi, clamp(phi, 0, 1), w, h, gc, x, y, g);
		}
		// Phi of (x, y, gamma_min) = 1
		write_data(Phi, 1.f, w, h, gc, x, y, 0);
		// Phi of (x, y, gamma_max) = 0
		write_data(Phi, 0.f, w, h, gc, x, y, gc - 1);
	}
}

/**
 * Update the P vector and save the result back to P
 *
 * x * y threads needed
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

	if (x < w && y < h)
	{
		// p has 3 channels
		int pc = 3;
		// maximum z index for P and Grad3_Phi
		int max_z = 3 * gc;
		int idx_z;

		float p, p_next, grad3_phi;

		for (int g = 0; g < gc; g++)
		{
			for (int i = 0; i < pc; i++)
			{
				idx_z = i * gc + g;
				p = read_data(P, w, h, max_z, x, y, idx_z);
				grad3_phi = read_data(Grad3_Phi, w, h, max_z, x, y, idx_z);

				p_next = p + tau_d * grad3_phi;

				// Write back to P
				write_data(P, p_next, w, h, max_z, x, y, idx_z);
			}
		}
	}
}

/**
 * Update the Phi vector and save the result back to Phi
 *
 * x * y threads needed
 *
 * Phi and Div3_P are dimensions (w * h * gc)
 */
__global__ void g_update_phi(float *Phi, float *Div3_P, int w, int h, int gc,
		float tau_p)
{
	// phi^k+1 = PD(phi^k + div3(p^k))
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float phi, phi_next, div3_p;

		for (int g = 0; g < gc; g++)
		{
			phi = read_data(Phi, w, h, gc, x, y, g);
			div3_p = read_data(Div3_P, w, h, gc, x, y, g);
			phi_next = phi + tau_p * div3_p;

			// Write back to Phi
			write_data(Phi, phi_next, w, h, gc, x, y, g);
		}
	}
}

/**
 * Calculate the gradient in x, y and gamma direction
 * Phi is dimensions (w * h * gc)
 *
 * x * y threads needed
 *
 * Grad3_Phi is the resulting w * h * gc * 3 with one channel for x, y and g direction
 */
__global__ void g_grad3(float *Phi, float *Grad3_Phi, int w, int h, int gc)
{
	// Gradient 3 is defined via forward differences
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float next_g = read_data(Phi, w, h, gc, x, y, gc);
		for (int g = gc; g > 0; g--)
		{
			float phi = read_data(Phi, w, h, gc, x, y, g);
			float dx = read_data(Phi, w, h, gc, x + 1, y, g) - phi;
			float dy = read_data(Phi, w, h, gc, x, y + 1, g) - phi;
			float dg = next_g - phi;

			next_g = phi;

			// 3 channels on the gradient with the same layout as p
			int max_z = 3 * gc;
			int idx_phi_x = 0 * gc + g;
			int idx_phi_y = 1 * gc + g;
			int idx_phi_g = 2 * gc + g;

			// Write the forward differences in different directions stacked (dx, dy, dg) into phi
			write_data(Grad3_Phi, dx, w, h, max_z, x, y, idx_phi_x);
			write_data(Grad3_Phi, dy, w, h, max_z, x, y, idx_phi_y);
			write_data(Grad3_Phi, dg, w, h, max_z, x, y, idx_phi_g);
		}

	}
}

/**
 * Calculate the divergence of P
 *
 * x * y threads needed
 *
 * P is stored in (w * h * gc * 3), so (p1, p2, p3)
 * Div3_P has the same dimensions as Phi (w * h * gc)
 */
__global__ void g_div3(float *P, float *Div3_P, int w, int h, int gc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		int idx_p1_z, idx_p2_z, idx_p3_z, idx_p3_z_1, // Idx for p3 with z - 1
				max_z;

		float p1, p2, p3, div3_p;

		for (int g = gc; g > 0; g--)
		{
			// Calculate the indices for p1, p2, p3
			max_z = 3 * gc;
			idx_p1_z = 0 * gc + g;
			idx_p2_z = 1 * gc + g;
			idx_p3_z = 2 * gc + g;
			// create last index, that may only lie in the range of the p3 index, thus clamp manually
			idx_p3_z_1 = clamp(idx_p3_z - 1, 2 * gc, (3 * gc) - 1);

			p1 = read_data(P, w, h, max_z, x, y, idx_p1_z);
			p2 = read_data(P, w, h, max_z, x, y, idx_p2_z);
			p3 = read_data(P, w, h, max_z, x, y, idx_p3_z);

			// Divergence 3 is defined as the sum of backward differences
			div3_p = p1 - read_data(P, w, h, max_z, x - 1, y, idx_p1_z) + p2
					- read_data(P, w, h, max_z, x, y - 1, idx_p2_z) + p3
					- read_data(P, w, h, max_z, x, y, idx_p3_z_1);

			write_data(Div3_P, div3_p, w, h, gc, x, y, g);
		}
	}
}

/**
 * Calculate the function u(x) (=G) from Phi
 *
 * x * y threads neeeded
 */
__global__ void g_compute_g(float *Phi, float *G, int w, int h, int gamma_min,
		int gamma_max)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	float gamma = (float) gamma_min;
	int gc = gamma_max - gamma_min + 1;

	for (int g = 0; g < gc; g++)
	{
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
	int gc = gamma_max - gamma_min + 1;

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

	// allocate raw input image array
	float *imgInL = new float[(size_t) w * h * nc];
	float *imgInR = new float[(size_t) w * h * nc];

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t) w * h * nc];

	// Init raw input image array
	convert_mat_to_layered(imgInL, mInL);
	convert_mat_to_layered(imgInR, mInR);

	// Allocate memory on device for images
	size_t imgBytes = w * h * nc * sizeof(float);
	size_t outBytes = w * h * gc * sizeof(float);
	float *IL, *IR, *IO = NULL;
	cudaMalloc(&IL, imgBytes);
	CUDA_CHECK;
	cudaMalloc(&IR, imgBytes);
	CUDA_CHECK;
	cudaMalloc(&IO, outBytes);
	CUDA_CHECK;
	cudaMemset(IL, 0, imgBytes);
	CUDA_CHECK;
	cudaMemset(IR, 0, imgBytes);
	CUDA_CHECK;
	cudaMemset(IO, 0, outBytes);
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

	// create kernel dimensions
	dim3 block2D(128, 1);
	dim3 grid2D((w + block2D.x - 1) / block2D.x,
			(h + block2D.y - 1) / block2D.y);

	dim3 block3D(64, 2, 1);
	dim3 grid3D((w + block3D.x - 1) / block3D.x,
			(h + block3D.y - 1) / block3D.y, (gc + block3D.z - 1) / block3D.z);

	dim3 block1D(256);
	dim3 grid1DP((w * h * gc * 3 + block1D.x - 1) / block1D.x);
	dim3 grid1DPhi((w * h * gc + block1D.x - 1) / block1D.x);

	// Test if rho works as intended, output an image with the errors in brightness
	g_test_rho<<<grid2D, block2D>>>(IL, IR, IO, w, h, nc, gc, gamma_min,
			lambda);
	CUDA_CHECK;

	// show input image
	showImage("Input", mInL, 100, 100); // show at position (x_from_left=100,y_from_above=100)

	size_t nBytes = w * h * sizeof(float);
	std::vector<cv::Mat *> mOuts;
	for (int g = 0; g < gc; g++)
	{
		cv::Mat *tmp = new cv::Mat(h, w, CV_32FC1);
		cudaMemcpy(imgOut, &IO[g * w * h], nBytes, cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		convert_layered_to_mat(*tmp, imgOut);
		normalize(*tmp, *tmp, 0, 1, cv::NORM_MINMAX, CV_32FC1);
		std::stringstream ss;
		ss << "Output" << g;
		showImage(ss.str(), *tmp, 100 * g, 100);
		mOuts.push_back(tmp);
	}

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
