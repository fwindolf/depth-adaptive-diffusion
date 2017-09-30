#include "util.h"
#include "usage.h"
#include "image.h"
#include "gradient.h"
#include "reduce.h"
#include "kernels.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include "assert.h"
using namespace std;

void check_Phi(float * Phi, int w, int h, int gc)
{
	cudaDeviceSynchronize();

	float * phi_check = new float[w * h];
	for (int g = 0; g < gc; g++)
	{
		cudaMemcpy(phi_check, &Phi[g * w * h], w * h * sizeof(float),
				cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		for (int i = 0; i < w * h; i++)
		{
			if (g == 0)
			{
				assert(phi_check[i] == 1.f);
			}
			else if (g == gc)
			{
				assert(phi_check[i] == 0.f);
			}
			else
			{
				assert(phi_check[i] <= 1.f && phi_check[i] >= 0.f);
			}
		}
	}
}

void check_P(float * P, float *Rho, int w, int h, int gc)
{
	cudaDeviceSynchronize();

	float * p1_check = new float[w * h];
	float * p2_check = new float[w * h];
	float * p3_check = new float[w * h];
	float * rho = new float[w * h];

	for (int g = 0; g < gc; g++)
	{
		size_t ip1 = g * w * h;
		size_t ip2 = (1 * gc + g) * w * h;
		size_t ip3 = (2 * gc + g) * w * h;
		cudaMemcpy(p1_check, &P[ip1], w * h * sizeof(float),
				cudaMemcpyDeviceToHost);
		CUDA_CHECK;

		cudaMemcpy(p2_check, &P[ip2], w * h * sizeof(float),
				cudaMemcpyDeviceToHost);
		CUDA_CHECK;

		cudaMemcpy(p3_check, &P[ip3], w * h * sizeof(float),
				cudaMemcpyDeviceToHost);
		CUDA_CHECK;

		cudaMemcpy(rho, &Rho[g * w * h], w * h * sizeof(float),
				cudaMemcpyDeviceToHost);
		CUDA_CHECK;

		for (int i = 0; i < w * h; i++)
		{
			if (sqrtf(square(p1_check[i]) + square(p2_check[i])) > 1)
			{
				cout << "i: " << i << ", g: " << g << ": p1=" << p1_check[i]
						<< ", p2=" << p2_check[i] << endl;
			}
			if (fabs(p3_check[i]) > rho[i])
			{
				cout << "i: " << i << ", g: " << g << ": rho=" << rho[i]
						<< ", p3=" << p3_check[i] << endl;
			}
		}
	}
}

cv::Mat calculate_disparities(const config c, cv::Mat mDisparities)
{
	// define the range of gamma
	int gc = c.gamma_max - c.gamma_min;

	// image + 0 is left
	string imageL = c.image + "0.png";
	// image + 1 is right
	string imageR = c.image + "1.png";

	cv::Mat mInL = load_image(imageL, c.gray, c.max_w, c.max_h);
	cv::Mat mInR = load_image(imageR, c.gray, c.max_w, c.max_h);

	// Width, height and channels of image
	int w, h, nc;
	get_dimensions(mInL, mInR, w, h, nc);

	// Define output array, grayscale image of depth values
	cv::Mat mOut(h, w, CV_32FC1);

	// allocate raw input image array
	float *imgInL = new float[(size_t) w * h * nc];
	float *imgInR = new float[(size_t) w * h * nc];

	float *imgDisparities = new float[(size_t) w * h];

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t) w * h * mOut.channels()];

	// Init raw input image array
	convert_mat_to_layered(imgInL, mInL);
	convert_mat_to_layered(imgInR, mInR);

	convert_mat_to_layered(imgDisparities, mDisparities);

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

	// for the error values Rho
	float * Rho = NULL;
	size_t rhoBytes = w * h * gc * sizeof(float);
	cudaMalloc(&Rho, rhoBytes);
	CUDA_CHECK;
	cudaMemset(Rho, 0, rhoBytes);
	CUDA_CHECK;

	// for the final depth values
	float * U = NULL;
	size_t uBytes = w * h * sizeof(float);
	cudaMalloc(&U, uBytes);
	CUDA_CHECK;
	cudaMemset(U, 0, uBytes);
	CUDA_CHECK;

	float * energy = NULL;
	cudaMalloc(&energy, sizeof(float));
	CUDA_CHECK;
	cudaMemset(energy, 0, sizeof(float));
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
	// Initialize Phi
	cudaMemcpy(U, imgDisparities, uBytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;

	g_init_phi<<<grid2D, block2D>>>(Phi, U, w, h, gc);
	CUDA_CHECK;

	cudaMemset(U, 0, uBytes);
	CUDA_CHECK;

	check_Phi(Phi, w, h, gc);

	for (int g = 0; g < gc; g++)
	{
		stringstream path;
		path << "phi/phi_00000_" << setfill('0') << setw(3) << g + c.gamma_min;
		save_from_GPU(path.str(), &Phi[g * w * h], w, h);
	}

	// Compute a global rho (that doesn't change...)
	g_compute_rho<<<grid2D, block2D>>>(IL, IR, Rho, w, h, nc, c.gamma_min,
			c.gamma_max, c.lambda);
	CUDA_CHECK;

	for (int g = 0; g < gc; g++)
	{
		stringstream path;
		path << "rho/rho_" << setfill('0') << setw(3) << g + c.gamma_min;
		save_from_GPU(path.str(), &Rho[g * w * h], w, h);
	}

	// Iterate until stopping criterion is reached
	int iterations = 1;
	while (1)
	{
		// Reset gradient and divergence
		cudaMemset(Grad3_Phi, 0, pBytes);
		CUDA_CHECK;
		cudaMemset(Div3_P, 0, phiBytes);
		CUDA_CHECK;

		// Calculate the divergence of P for the update step of phi
		g_div3<<<grid2D, block2D>>>(P, Div3_P, w, h, gc, c.dx, c.dy, c.dg);

		// Update the Phi
		g_update_phi<<<grid2D, block2D>>>(Phi, Div3_P, w, h, gc, c.tau_p);
		CUDA_CHECK;

		check_Phi(Phi, w, h, gc);

		// Calculate the gradient in x, y, and gamma direction
		g_grad3<<<grid2D, block2D>>>(Phi, Grad3_Phi, w, h, gc, c.dx, c.dy,
				c.dg);
		CUDA_CHECK;

		// Update the P
		g_update_p<<<grid2D, block2D>>>(P, Grad3_Phi, Rho, w, h, gc, c.tau_d);
		CUDA_CHECK;

		// check convergence
		if (iterations % (c.max_iterations / 2) == 0)
		{
			cout << "Iteration " << iterations << endl;
			/*
			 for (int g = 0; g < gc; g++)
			 {
			 stringstream path1, path2, path3;
			 path1 << "p/p1_" << setfill('0') << setw(5) <<  iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 path2 << "p/p2_" << setfill('0') << setw(5) <<  iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 path3 << "p/p3_" << setfill('0') << setw(5) << 	 iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 size_t ip1 = g * w * h;
			 size_t ip2 = (1 * gc + g) * w * h;
			 size_t ip3 = (2 * gc + g) * w * h;
			 save_from_GPU(path1.str(), &P[ip1], w, h);
			 save_from_GPU(path2.str(), &P[ip2], w, h);
			 save_from_GPU(path3.str(), &P[ip3], w, h);
			 }

			 for (int g = 0; g < gc; g++)
			 {
			 stringstream pathx, pathy, pathg;
			 pathx << "grad/gradx_" << setfill('0') << setw(5) <<  iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 pathy << "grad/grady_" << setfill('0') << setw(5) <<  iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 pathg << "grad/gradg_" << setfill('0') << setw(5) <<  iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 size_t ip1 = g * w * h;
			 size_t ip2 = (1 * gc + g) * w * h;
			 size_t ip3 = (2 * gc + g) * w * h;
			 save_from_GPU(pathx.str(), &Grad3_Phi[ip1], w, h);
			 save_from_GPU(pathy.str(), &Grad3_Phi[ip2], w, h);
			 save_from_GPU(pathg.str(), &Grad3_Phi[ip3], w, h);
			 }
			 */
			 for (int g = 0; g < gc; g++)
			 {
			 stringstream path;
			 path << "phi/phi_" << setfill('0') << setw(5) << iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 save_from_GPU(path.str(), &Phi[g * w * h], w, h);
			 }
			 /*
			 for (int g = 0; g < gc; g++)
			 {
			 stringstream path;
			 path << "div/div_" << setfill('0') << setw(5) << iterations << "_" << setfill('0') << setw(3) << g + c.gamma_min;
			 save_from_GPU(path.str(), &Div3_P[g * w * h], w, h);
			 }
			 */

			// Calculate the new G
			g_compute_u<<<grid2D, block2D>>>(Phi, U, w, h, c.gamma_min,
					c.gamma_max);
			CUDA_CHECK;

			cudaMemset(energy, 0, sizeof(float));
			CUDA_CHECK;

			g_compute_energy<<<grid2D, block2D>>>(U, IL, IR, energy, w, h, nc,
					c.lambda);
			CUDA_CHECK;

			float energy_host = 0.f;
			cudaMemcpy(&energy_host, energy, sizeof(float),
					cudaMemcpyDeviceToHost);
			CUDA_CHECK;

			cout << iterations << ": Energy is " << energy_host << endl;

			if (energy_host < 0.01 || iterations >= c.max_iterations)
				break;
		}

		iterations++;
	}

	// Move disparities from device to host
	cudaMemcpy(imgOut, U, uBytes, cudaMemcpyDeviceToHost);
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
	cudaFree(U);
	CUDA_CHECK;

	return mOut;
}

cv::Mat adaptive_diffusion(const cv::Mat mDisparities, const cv::Mat mIn,
		const config c)
{
	cout << "Depth Adaptive Diffusion" << endl;

	// Width, height and channels of image
	int w, h, nc;
	get_dimensions(mIn, w, h, nc);

	cv::Mat mDiffused = cv::Mat(h, w, CV_32FC3);

	// Convert to layered representation
	float *imgIn = new float[(size_t) w * h * nc];
	convert_mat_to_layered(imgIn, mIn);

	float *imgDisparities = new float[(size_t) w * h];
	convert_mat_to_layered(imgDisparities, mDisparities);

	// Input, Disparities, Output image
	float *In, *Out = NULL;
	size_t nbytes = (size_t) (w * h * nc) * sizeof(float);

	float *Disparities, *Depths = NULL;
	size_t ndisparities = (size_t) (w * h) * sizeof(float);
	size_t ndepths = (size_t) (w * h) * sizeof(float);

	// G matrix
	float *G = NULL;
	size_t ngbytes = (size_t) (w * h) * sizeof(float);

	// Gradient in x,y direction, divergence
	float *Grad_x, *Grad_y, *Divergence = NULL;

	// Reserve space on device
	cudaMalloc(&In, nbytes);
	CUDA_CHECK;
	cudaMalloc(&Out, nbytes);
	CUDA_CHECK;
	cudaMalloc(&Grad_x, nbytes);
	CUDA_CHECK;
	cudaMalloc(&Grad_y, nbytes);
	CUDA_CHECK;
	cudaMalloc(&Divergence, nbytes);
	CUDA_CHECK;
	cudaMalloc(&Disparities, ndisparities);
	CUDA_CHECK;
	cudaMalloc(&Depths, ndepths);
	CUDA_CHECK;
	cudaMalloc(&G, ngbytes);
	CUDA_CHECK;

	cudaMemset(In, 0, nbytes);
	CUDA_CHECK;
	cudaMemset(Out, 0, nbytes);
	CUDA_CHECK;
	cudaMemset(G, 0, ngbytes);
	CUDA_CHECK;

	// Copy disparities to device
	cudaMemcpy(Disparities, imgDisparities, ndisparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// Copy image to device
	cudaMemcpy(In, imgIn, nbytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;

	dim3 block2D(128, 1);
	dim3 grid2D((w + block2D.x - 1) / block2D.x,
			(h + block2D.y - 1) / block2D.y);

	dim3 block3D(64, 2, nc);
	dim3 grid3D((w + block3D.x - 1) / block3D.x,
			(h + block3D.y - 1) / block3D.y, (nc + block3D.z - 1) / block3D.z);

	// Compute the depth from the disparity values
	g_compute_depth<<<grid2D, block2D>>>(Disparities, Depths, w, h, c.baseline,
			c.focal_length, c.doffs);
	CUDA_CHECK;

	// Normalize to [0, 1]
	normalize(Depths, w, h, 0.f, 1.f);

	// ---- Calculate the G matrix
	g_compute_g_matrix<<<grid2D, block2D>>>(Depths, G, w, h, c.focal_plane,
			c.radius);
	CUDA_CHECK;

	// Normalize to [0, 1]
	normalize(G, w, h, 0.f, 1.f);

	save_from_GPU("depths", Depths, w, h);
	save_from_GPU("g", G, w, h);

	for (int i = 0; i < 15; i++)
	{
		// reset the gradient/divergence data
		cudaMemset(Grad_x, 0, nbytes);
		CUDA_CHECK;
		cudaMemset(Grad_y, 0, nbytes);
		CUDA_CHECK;
		cudaMemset(Divergence, 0, nbytes);
		CUDA_CHECK;

		//-- Gradient
		g_gradient<<<grid3D, block3D>>>(In, Grad_x, Grad_y, w, h, nc);
		CUDA_CHECK;

		// apply G matrix from texture memory
		g_apply_g<<<grid2D, block2D>>>(Grad_x, Grad_y, G, w, h, nc);
		CUDA_CHECK;

		// calculate divergence
		g_divergence<<<grid3D, block3D>>>(Grad_x, Grad_y, Divergence, w, h, nc);
		CUDA_CHECK;

		// do the update step
		g_update_step<<<grid3D, block3D>>>(In, Divergence, w, h, nc, c.tau);
		CUDA_CHECK;
	}

	// Copy result back to host
	cudaMemcpy(imgIn, In, nbytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	convert_layered_to_mat(mDiffused, imgIn);

	// free memory
	cudaFree(In);
	CUDA_CHECK;
	cudaFree(G);
	CUDA_CHECK;
	cudaFree(Grad_x);
	CUDA_CHECK;
	cudaFree(Grad_y);
	CUDA_CHECK;
	cudaFree(Divergence);
	CUDA_CHECK;
	cudaFree(Disparities);
	CUDA_CHECK;
	cudaFree(Depths);
	CUDA_CHECK;

	delete[] imgIn;
	delete[] imgDisparities;

	return mDiffused;
}

int main(int argc, char **argv)
{
	init_device();

	// Create empty config
	config c;

	read_parameters(c, argc, argv);

	// Load input images
	std::string imageL = c.image + "0.png", imageR = c.image + "1.png";

	cv::Mat mInL = load_image(imageL, c.gray, c.max_w, c.max_h);
	cv::Mat mInR = load_image(imageR, c.gray, c.max_w, c.max_h);

	int w, h, nc;
	get_dimensions(mInL, w, h, nc);

	cout << "Image Dimensions: " << w << "x" << h << "x" << nc << endl;

	// Get disparities from dataset or calculate
	cv::Mat mDisparities;
	if (c.disparities_from_file)
	{
		mDisparities = load_pfm(c.disparities, c.max_w, c.max_h);
	}
	else
	{
		cerr
				<< "ERROR: Call without ground truth not supported in this version!"
				<< endl;
		exit(1);
	}
	mDisparities = calculate_disparities(c, mDisparities);

	// Do anisotropic diffusion with the depth values
	cv::Mat mOut = adaptive_diffusion(mDisparities, mInL, c);

	showImage("Input", mInL, 100, 100);

	// Reduce range from [0, 255] to [0, 1]
	mDisparities /= 255.f;
	//normalize(mDisparities, mDisparities, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	showImage("Disparities", mDisparities, 600, 100);
	save_image("disparities", mDisparities);

	showImage("Output", mOut, 100, 600);
	save_image("out", mOut);
	// wait for key inputs
	cv::waitKey(0);

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}
