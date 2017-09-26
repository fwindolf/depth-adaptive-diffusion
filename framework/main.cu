#include "util.h"
#include "usage.h"
#include "image.h"
#include "kernels.h"
#include <iostream>
using namespace std;

cv::Mat calculate_disparities(const config c)
{
	// define the range of gamma
	int gc = c.gamma_max - c.gamma_min + 1;

	// image + 0 is left
	string imageL = c.image + "0.png";
	// image + 1 is right
	string imageR = c.image + "1.png";

	cv::Mat mInL = load_image(imageL, c.gray);
	cv::Mat mInR = load_image(imageR, c.gray);

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
	float * G, *G_last = NULL;
	cudaMalloc(&G, gBytes);
	CUDA_CHECK;
	cudaMemset(G, 0, gBytes);
	CUDA_CHECK;
	cudaMalloc(&G_last, gBytes);
	CUDA_CHECK;
	cudaMemset(G_last, 0, gBytes);
	CUDA_CHECK;

	float * err = NULL;
	cudaMalloc(&err, sizeof(float));
	CUDA_CHECK;
	cudaMemset(err, 0, sizeof(float));
	CUDA_CHECK;

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

	// Actual Algorithm
	// Initialization works with 0s for P, only for Phi the first layer needs to be 1s
	// Thus run the projection once
	g_project_phi_d<<<grid3D, block3D>>>(Phi, w, h, gc);
	CUDA_CHECK;

	// Iterate until stopping criterion is reached
	int iterations = 0;
	while (1)
	{
		// Reset gradient and divergence
		cudaMemset(Grad3_Phi, 0, pBytes);
		CUDA_CHECK;
		cudaMemset(Div3_P, 0, phiBytes);
		CUDA_CHECK;

		// Calculate the divergence of P for the update step of phi
		g_div3<<<grid2D, block2D>>>(P, Div3_P, w, h, gc);
		CUDA_CHECK;

		// Update the Phi
		g_update_phi<<<grid2D, block2D>>>(Phi, Div3_P, w, h, gc, c.tau_p);
		CUDA_CHECK;

		// Make sure Phi is in the solution space (D)
		g_project_phi_d<<<grid2D, block2D>>>(Phi, w, h, gc);
		CUDA_CHECK;

		// Calculate the gradient in x, y, and gamma direction
		g_grad3<<<grid2D, block2D>>>(Phi, Grad3_Phi, w, h, gc);
		CUDA_CHECK;

		// Update the P
		g_update_p<<<grid2D, block2D>>>(P, Grad3_Phi, w, h, gc, c.tau_d);
		CUDA_CHECK;

		// Make sure P is in solution space (C)
		g_project_p_c<<<grid2D, block2D>>>(P, IL, IR, w, h, nc, gc, c.lambda,
				c.gamma_min);
		CUDA_CHECK;

		if (iterations > c.max_iterations)
			break;

		// TODO: convergence check via energy that is minimized, not via change of g
		// check convergence
		if (iterations % 1000 == 0)
		{
			// Save G of last convergence check
			cudaMemcpy(G_last, G, gBytes, cudaMemcpyDeviceToDevice);
			CUDA_CHECK;

			// Calculate the new G
			g_compute_g<<<grid2D, block2D>>>(Phi, G, w, h, c.gamma_min,
					c.gamma_max);
			CUDA_CHECK;

			cudaMemset(err, 0, sizeof(float));
			CUDA_CHECK;

			g_squared_err_g<<<grid2D, block2D>>>(G, G_last, w, h, err);
			CUDA_CHECK;

			float err_host = 0.f;
			cudaMemcpy(&err_host, err, sizeof(float), cudaMemcpyDeviceToHost);
			CUDA_CHECK;

			cout << iterations << ": Error is " << err_host << endl;

			if (sqrt(err_host) < 0.01 || iterations > c.max_iterations)
				break;
		}

		iterations++;
	}

	// Move disparities from device to host
	cudaMemcpy(imgOut, G, gBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// show output image: first convert to interleaved opencv format from the layered raw array
	convert_layered_to_mat(mOut, imgOut);

	// free allocated arrays
	delete[] imgInL;
	delete[] imgInR;
	delete[] imgOut;

	cudaFree(IL);
	CUDA_CHECK;
	cudaFree(IR);
	CUDA_CHECK;
	cudaFree(P);
	CUDA_CHECK;
	cudaFree(Phi);
	CUDA_CHECK;
	cudaFree(Grad3_Phi);
	CUDA_CHECK;
	cudaFree(Div3_P);
	CUDA_CHECK;
	cudaFree(G);
	CUDA_CHECK;
	cudaFree(G_last);
	CUDA_CHECK;
	cudaFree(err);
	CUDA_CHECK;

	return mOut;
}

cv::Mat adaptive_diffusion(const cv::Mat mDisparities, const config c)
{
	cv::Mat mDiffused;



	return mDiffused;
}

int main(int argc, char **argv)
{
	init_device();

	// Create empty config
	config c;

	read_parameters(c, argc, argv);

	// Load input images
	std::string imageL = c.image + "0.png",
				imageR = c.image + "1.png";

	cv::Mat mInL = load_image(imageL, c.gray);
	cv::Mat mInR = load_image(imageR, c.gray);

	// Get disparities from dataset or calculate
	cv::Mat mDisparities;
	if (c.disparities_from_file)
	{
		mDisparities = load_pfm(c.disparities);
	}
	else
	{
		mDisparities = calculate_disparities(c);
	}


	showImage("Input", mInL, 100, 100);
	// normalize(mDisparities, mDisparities, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	showImage("Disparities", mDisparities, 500, 100);

	// wait for key inputs
	cv::waitKey(0);

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}
