#include "util.h"
#include "usage.h"
#include "image.h"
#include "kernels.h"
#include <iostream>
using namespace std;

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
		g_update_phi<<<grid2D, block2D>>>(Phi, Div3_P, w, h, gc, tau_p);
		CUDA_CHECK;

		// Make sure Phi is in the solution space (D)
		g_project_phi_d<<<grid2D, block2D>>>(Phi, w, h, gc);
		CUDA_CHECK;

		// Calculate the gradient in x, y, and gamma direction
		g_grad3<<<grid2D, block2D>>>(Phi, Grad3_Phi, w, h, gc);
		CUDA_CHECK;

		// Update the P
		g_update_p<<<grid2D, block2D>>>(P, Grad3_Phi, w, h, gc, tau_d);
		CUDA_CHECK;

		// Make sure P is in solution space (C)
		g_project_p_c<<<grid2D, block2D>>>(P, IL, IR, w, h, nc, gc, lambda,
				gamma_min);
		CUDA_CHECK;

		if (iterations > max_iterations)
			break;

		iterations++;
	}

	// Calculate the new G
	g_compute_g<<<grid2D, block2D>>>(Phi, G, w, h, gamma_min, gamma_max);

	/*
	// Visualization only makes sense in that way if gamma = -1 .. 1
	float * imDiv3 = new float[w * h * gc];
	float * imGrad31 = new float[w * h * gc];
	float * imGrad32 = new float[w * h * gc];
	float * imGrad33 = new float[w * h * gc];

	cv::Mat mDiv3(h, w, CV_32FC3);
	cv::Mat mGrad31(h, w, CV_32FC3);
	cv::Mat mGrad32(h, w, CV_32FC3);
	cv::Mat mGrad33(h, w, CV_32FC3);

	cudaMemcpy(imDiv3, Div3_P, phiBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(imGrad31, &Grad3_Phi[0], phiBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(imGrad31, &Grad3_Phi[w * h * gc], phiBytes,
			cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(imGrad31, &Grad3_Phi[2 * w * h * gc], phiBytes,
			cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	convert_layered_to_mat(mDiv3, imDiv3);
	convert_layered_to_mat(mGrad31, imGrad31);
	convert_layered_to_mat(mGrad32, imGrad32);
	convert_layered_to_mat(mGrad33, imGrad33);

	normalize(mDiv3, mDiv3, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	normalize(mGrad31, mGrad31, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	normalize(mGrad32, mGrad32, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	normalize(mGrad33, mGrad33, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);

	showImage("Div3", mDiv3, 100, 100 + h + 40);
	showImage("Grad31", mGrad31, 400, 100 + h + 40);
	showImage("Grad32", mGrad32, 700, 100 + h + 40);
	showImage("Grad33", mGrad33, 1000, 100 + h + 40);

	float * imPhi = new float[w * h * gc];
	float * imP1 = new float[w * h * gc];
	float * imP2 = new float[w * h * gc];
	float * imP3 = new float[w * h * gc];

	cv::Mat mPhi(h, w, CV_32FC3);
	cv::Mat mP1(h, w, CV_32FC3);
	cv::Mat mP2(h, w, CV_32FC3);
	cv::Mat mP3(h, w, CV_32FC3);

	cudaMemcpy(imPhi, Phi, phiBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(imP1, &P[0], phiBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(imP1, &P[w * h * gc], phiBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(imP1, &P[2 * w * h * gc], phiBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	convert_layered_to_mat(mPhi, imPhi);
	convert_layered_to_mat(mP1, imP1);
	convert_layered_to_mat(mP2, imP2);
	convert_layered_to_mat(mP3, imP3);

	normalize(mPhi, mPhi, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	normalize(mP1, mP1, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	normalize(mP2, mP2, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	normalize(mP3, mP3, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);

	showImage("Phi", mPhi, 1100, 100);
	showImage("P1", mP1, 1400, 100);
	showImage("P2", mP2, 1700, 100);
	showImage("P3", mP3, 2000, 100);
	*/

	// show input image
	showImage("Input", mInL, 100, 100); // show at position (x_from_left=100,y_from_above=100)

	// Move disparities from device to host
	cudaMemcpy(imgOut, G, gBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

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
