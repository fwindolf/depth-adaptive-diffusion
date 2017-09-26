#include "util.h"
#include "usage.h"
#include "image.h"
#include "gradient.h"
#include "kernels.h"
#include <iostream>
using namespace std;

// Texture memory for the G values
texture<float, 2, cudaReadModeElementType> texRef;

/**
 * Calculate the depth from the disparity values
 *
 * Z = baseline * f / (d + doffs)
 * baseline: 	camera baseline in mm
 * f: 			focal length in pixels
 * d:			disparity for pixel
 * doffs:		x-difference of principal points (cx1 - cx0) for im1 and im0
 */
__global__ void g_compute_depth(float * Disparities, float *Depths, int w,
		int h, float baseline, int f, int doffs)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float d = read_data(Disparities, w, h, x, y);
		write_data(Depths, baseline * f / (d + doffs), w, h, x, y);
	}
}

/**
 * Compute the alpha parameter for the circle of confusion
 *
 * alpha = F²/(n * (Zf - F)) with focal length F and aperture n
 * and depth of the focal plane between gamma_min, gamma_max = [0, 1]
 */
__device__ __host__ float alpha(float f, float n, int z_f)
{
	return square(f) / (n * (z_f - f));
}

/**
 * Compute the g function via an estimated circle of confusion
 *
 * See: Bertalmio, Fort et al: Real-time, Accurate Depth of Field
 * using Anisotropic Diffusion and Programmable Graphic Cards
 * http://www.dtic.upf.edu/~mbertalmio/dof/dof01.pdf
 *
 * g(x,y) = alpha * (| Z(x,y) - Zf | / Z(x,y) )²
 *
 * z_f is the depth of the focal plane between gamma_min, gamma_max = [0,1]
 * g needs to be scaled in order to lie between [0,1]
 */
__global__ void g_compute_g(float *Depths, float *G, int w, int h, float z_f,
		float alpha)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	float z;

	if (x < w && y < h)
	{
		z = read_data(Depths, w, h, x, y);
		write_data(G, alpha * square(fabs(z - z_f) / z), w, h, x, y);
	}
}

/**
 * Apply G and write the result back to the gradient vectors
 */
__global__ void g_apply_g(float *Grad_x, float *Grad_y, int w, int h, int nc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float g = tex2D(texRef, x + 0.5f, y + 0.5f);
		float grad_x, grad_y;

		for (int c = 0; c < nc; c++)
		{
			grad_x = read_data(Grad_x, w, h, nc, x, y, c);
			grad_y = read_data(Grad_y, w, h, nc, x, y, c);

			// Write v_1 back to V_1 and v_2 to V_2
			write_data(Grad_x, g * grad_x, w, h, nc, x, y, c);
			write_data(Grad_y, g * grad_y, w, h, nc, x, y, c);
		}
	}

}

/**
 * Compute the update step as In+1 = In + tau*D
 */
__global__ void g_update_step(float *D, float *I, int w, int h, int nc,
		float tau)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z + blockDim.z * blockIdx.z;

	float upd;

	if (x < w && y < h && c < nc)
	{
		upd = read_data(I, w, h, nc, x, y, c)
				+ tau * read_data(D, w, h, nc, x, y, c);
		write_data(I, upd, w, h, nc, x, y, c);
	}
}


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


cv::Mat adaptive_diffusion(const cv::Mat mDisparities, const cv::Mat mIn,
		const config c)
{
	cv::Mat mDiffused;

	// Width, height and channels of image
	int w, h, nc;
	get_dimensions(mIn, w, h, nc);

	// Convert to layered representation
	float *imgIn = new float[(size_t) w * h * nc];
	convert_mat_to_layered(imgIn, mIn);

	float *imgDisparities = new float[(size_t) w * h];
	convert_mat_to_layered(imgDisparities, mDisparities);

	// Input, Disparities, Output image
	float *In, *Disparities, *Depths, *Out = NULL;

	// Gradient in x,y direction, divergence
	float *Grad_x, *Grad_y, *Divergence = NULL;

	size_t nbytes = (size_t) (w * h * nc) * sizeof(float);
	size_t ndisparities = (size_t) (w * h) * sizeof(int);
	size_t ndepths = (size_t) (w * h) * sizeof(float);

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

	cudaMemset(In, 0, nbytes);
	CUDA_CHECK;
	cudaMemset(Out, 0, nbytes);
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

	dim3 block3D(64, 2, 1);
	dim3 grid3D((w + block3D.x - 1) / block3D.x,
			(h + block3D.y - 1) / block3D.y, (nc + block3D.z - 1) / block3D.z);

	// Compute the depth from the disparity values
	g_compute_depth<<<grid2D, block2D>>>(Disparities, Depths, w, h, c.baseline, c.focal_length, c.doffs);

	// ---- Calculate the G matrix
	float a = alpha(c.focal_length, 1.f, c.focal_plane);

	// Setup disparities as texture memory
	texRef.addressMode[0] = cudaAddressModeClamp;// clamp x to border
	texRef.addressMode[1] = cudaAddressModeClamp; // clamp y to border
	texRef.filterMode = cudaFilterModeLinear; // linear interpolation
	texRef.normalized = false; // access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, &texRef, Disparities, &desc, w, h, sizeof(float));

	for (int i = 0; i < c.max_iterations; i++)
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
		g_apply_g<<<grid2D, block2D>>>(Grad_x, Grad_y, w, h, nc);
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

	// Do anisotropic diffusion with the depth values
	cv::Mat mOut = adaptive_diffusion(mDisparities, mInL, c);

	showImage("Input", mInL, 100, 100);
	// normalize(mDisparities, mDisparities, 0.f, 1.f, cv::NORM_MINMAX, CV_32FC1);
	showImage("Disparities", mDisparities, 500, 100);

	showImage("Output", mOut, 100, 500);

	// wait for key inputs
	cv::waitKey(0);

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}
