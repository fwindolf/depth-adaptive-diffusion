#include "kernels.h"
#include "util.h"

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

			float r = rho(iL, iR, nc, lambda);
			p3 = p3 / max(1.f, fabs(p3) / r);

			// write the results back to P
			write_data(P, p1, w, h, max_z, x, y, idx_p1_z);
			write_data(P, p2, w, h, max_z, x, y, idx_p2_z);
			write_data(P, p3, w, h, max_z, x, y, idx_p3_z);
		}
	}
}

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

__global__ void g_grad3(float *Phi, float *Grad3_Phi, int w, int h, int gc)
{
	// Gradient 3 is defined via forward differences
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float next_g = read_data(Phi, w, h, gc, x, y, gc);
		// Run from gamma_max (gc - 1) to gamma_min (0)
		for (int g = gc - 1; g >= 0; g--)
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

__global__ void g_div3(float *P, float *Div3_P, int w, int h, int gc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		int idx_p1_z, idx_p2_z, idx_p3_z, idx_p3_z_1, // Idx for p3 with z - 1
				max_z;

		float p1, p2, p3, div3_p;

		for (int g = gc - 1; g > 0; g--)
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

__global__ void g_compute_g(float *Phi, float *G, int w, int h, int gamma_min,
		int gamma_max)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int gamma = gamma_min;
	int gc = gamma_max - gamma_min;

	if (x < w && y < h)
	{
		for (int g = 0; g < gc; g++)
		{
			// use mu = 0.5 aka round to nearest integer value
			gamma += (int) round(read_data(Phi, w, h, gc, x, y, g));
		}

		write_data(G, gamma, w, h, x, y);
	}
}

__global__ void g_compute_energy(float * G, float *IL, float *IR, float * energy, int w, int h, int nc, float lambda)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	__shared__ float sm;

	if (x < w && y < h)
	{
		float e = 0.f;

		// Regularizing term: |grad(u(x))|
		int g = read_data(G, w, h, x, y);
		int gx = read_data(G, w, h, x + 1, y);
		int gy = read_data(G, w, h, x, y + 1);
		e += sqrt(square(gx - g) + square(gy - g)); // Regularizing term

		// Data term: rho(u(x), x)
		float iL[3];
		float iR[3];

		// Save image data to temporary arrays
		for (int c = 0; c < nc; c++)
		{
			iL[c] = read_data(IL, w, h, nc, x, y, c);
			iR[c] = read_data(IR, w, h, nc, x + g, y, c); // g is the current, calculated disparity value
		}
		e += rho(iL, iR, nc, lambda);

		atomicAdd(&sm, e);
		__syncthreads();


		// Add this to the current energy
		if(threadIdx.x == 0)
			atomicAdd(energy, sm);
	}

}
