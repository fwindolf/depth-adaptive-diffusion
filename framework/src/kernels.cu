#include "kernels.h"
#include "util.h"
#include "assert.h"

// TODO: Use shared memory to load data from iL and iR to shared memory
__global__ void g_compute_rho(float *iL, float *iR, float *Rho, int w, int h,
		int nc, int gamma_min, int gamma_max, float lambda, float dg)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		// Iterate all possible disparities
		for (int g = gamma_min; g < gamma_max; g++)
		{
			float r = 0.f;

			// Calculate absolute error between iL and iR
			for (int c = 0; c < nc; c++)
			{
				float il = read_data(iL, w, h, nc, x, y, c);
				float ir = read_data(iR, w, h, nc, x - (g * dg), y, c);
				r += lambda * fabs(il - ir);
			}
			// Create entry at layer g (normalized to range from 0 to gamma_max - gamma_min)
			int gc = gamma_max - gamma_min;
			write_data(Rho, r, w, h, gc, x, y, g - gamma_min);
		}
	}
}

__global__ void g_init_phi(float *Phi, int w, int h, int gc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		// Initialize gamma_min to 1
		write_data(Phi, 1.f, w, h, gc, x, y, 0);
	}
}

__global__ void g_update_phi(float *Phi, float *Div3_P, int w, int h, int gc,
		float tau_p)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		// Only do this for the layers that don't get set anyways
		for (int g = 1; g < gc - 1; g++)
		{
			float phi = read_data(Phi, w, h, gc, x, y, g);
			float div3_p = read_data(Div3_P, w, h, gc, x, y, g);
			float upd = fclamp(phi + tau_p * div3_p, 0.f, 1.f);
			write_data(Phi, upd, w, h, gc, x, y, g);
		}
		// Set phi(x, gamma_min) = 1 and phi(x, gamma_max) = 0
		write_data(Phi, 1.f, w, h, gc, x, y, 0);
		write_data(Phi, 0.f, w, h, gc, x, y, gc - 1);
	}
}

__global__ void g_update_p(float *P, float *Grad3_Phi, float *Rho, int w, int h,
		int gc, float tau_d)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		int max_z, idx_p1_z, idx_p2_z, idx_p3_z;
		float p1, p2, p3, g1, g2, g3, pabs, r;

		for (int g = 0; g < gc; g++)
		{
			max_z = 3 * gc;
			idx_p1_z = 0 * gc + g;
			idx_p2_z = 1 * gc + g;
			idx_p3_z = 2 * gc + g;

			p1 = read_data(P, w, h, max_z, x, y, idx_p1_z);
			p2 = read_data(P, w, h, max_z, x, y, idx_p2_z);
			p3 = read_data(P, w, h, max_z, x, y, idx_p3_z);

			g1 = read_data(Grad3_Phi, w, h, max_z, x, y, idx_p1_z);
			g2 = read_data(Grad3_Phi, w, h, max_z, x, y, idx_p2_z);
			g3 = read_data(Grad3_Phi, w, h, max_z, x, y, idx_p3_z);

			// Update
			p1 = p1 + tau_d * g1;
			p2 = p2 + tau_d * g2;
			p3 = p3 + tau_d * g3;

			// Project p1 and p2 to C
			pabs = sqrt(square(p1) + square(p2));
			if (pabs > 1.f)
			{
				p1 = p1 / pabs;
				p2 = p2 / pabs;
			}

			r = read_data(Rho, w, h, gc, x, y, g);
			if (fabs(p3) > r)
				p3 = copysignf(r, p3); // r with sign of p3

			write_data(P, p1, w, h, max_z, x, y, idx_p1_z);
			write_data(P, p2, w, h, max_z, x, y, idx_p2_z);
			write_data(P, p3, w, h, max_z, x, y, idx_p3_z);
		}
	}
}

__global__ void g_grad3(float *Phi, float *Grad3_Phi, int w, int h, int gc,
		float dx, float dy, float dg)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		for (int g = 0; g < gc; g++)
		{
			int max_z = 3 * gc;
			int idx_p1_z = 0 * gc + g;
			int idx_p2_z = 1 * gc + g;
			int idx_p3_z = 2 * gc + g;

			float phi = read_data(Phi, w, h, gc, x, y, g);
			float grad_x = (read_data(Phi, w, h, gc, x + 1, y, g) - phi) / dx;
			float grad_y = (read_data(Phi, w, h, gc, x, y + 1, g) - phi) / dy;
			float grad_g = (read_data(Phi, w, h, gc, x, y, g + 1) - phi) / dg;

			write_data(Grad3_Phi, grad_x, w, h, max_z, x, y, idx_p1_z);
			write_data(Grad3_Phi, grad_y, w, h, max_z, x, y, idx_p2_z);
			write_data(Grad3_Phi, grad_g, w, h, max_z, x, y, idx_p3_z);
		}
	}
}

__global__ void g_div3(float *P, float *Div3_P, int w, int h, int gc, float dx,
		float dy, float dg)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		for (int g = 0; g < gc; g++)
		{
			int max_z = 3 * gc;
			int idx_p1_z = 0 * gc + g;
			int idx_p2_z = 1 * gc + g;
			int idx_p3_z = 2 * gc + g;

			float div3 = 0.f;
			float p1 = read_data(P, w, h, max_z, x, y, idx_p1_z);
			float p2 = read_data(P, w, h, max_z, x, y, idx_p2_z);
			float p3 = read_data(P, w, h, max_z, x, y, idx_p3_z);

			div3 += (p1 - read_data(P, w, h, max_z, x - 1, y, idx_p1_z)) / dx;
			div3 += (p2 - read_data(P, w, h, max_z, x, y - 1, idx_p2_z)) / dy;
			// Make sure p3(x, y, g - 1) does not reach into p2 values, else div(p3) is 0 anyways
			if (g > 0)
				div3 += (p3 - read_data(P, w, h, max_z, x, y, idx_p3_z - 1))
						/ dg;

			write_data(Div3_P, div3, w, h, gc, x, y, g);
		}
	}
}

__global__ void g_compute_u(float *Phi, float *U, int w, int h, int gamma_min,
		int gamma_max)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float u = gamma_min;
		for (int g = 0; g < gamma_max - gamma_min; g++)
		{
			u += read_data(Phi, w, h, gamma_max - gamma_min, x, y, g);
		}
		write_data(U, u, w, h, x, y);
	}
}

__global__ void g_compute_energy(float *Grad3_Phi, float *Phi, float *Rho,
		float *energy, int w, int h, int gc, float lambda)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	__shared__ float sm;
	if (x < w && y < h)
	{
		float e = 0.f;
		for (int g = 0; g < gc; g++)
		{
			float rho = read_data(Rho, w, h, gc, x, y, g);
			int idx_x = g;
			int idx_y = gc + g;
			int idx_g = 2 * gc + g;

			// Calculate the lifted energy E = |grad(phi(x, gamma))| + rho(x, gamma)*|grad_g(phi(x, gamma))|
			float grad_x = read_data(Grad3_Phi, w, h, (3 * gc), x, y, idx_x);
			float grad_y = read_data(Grad3_Phi, w, h, (3 * gc), x, y, idx_y);
			float grad_g = read_data(Grad3_Phi, w, h, (3 * gc), x, y, idx_g);

			e += sqrtf(square(grad_x) + square(grad_y)) + (rho * fabs(grad_g));
		}

		// Add up the energy of this block
		atomicAdd(&sm, e);
		__syncthreads();

		// Add this to the current (global) energy
		if (threadIdx.x == 0)
			atomicAdd(energy, sm);

	}
}

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

__global__ void g_compute_g_matrix(float *Depths, float *G, int w, int h,
		float z_f, float radius)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float z = read_data(Depths, w, h, x, y);
		write_data(G, powf(fabs(z - z_f), radius), w, h, x, y);
	}
}

__global__ void g_apply_g(float *Grad_x, float *Grad_y, float *G, int w, int h,
		int nc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h)
	{
		float g = read_data(G, w, h, x, y);
		float gx, gy;

		for (int c = 0; c < nc; c++)
		{
			gx = read_data(Grad_x, w, h, nc, x, y, c);
			gy = read_data(Grad_y, w, h, nc, x, y, c);

			write_data(Grad_x, gx * g, w, h, nc, x, y, c);
			write_data(Grad_y, gy * g, w, h, nc, x, y, c);
		}
	}
}

__global__ void g_update_step(float *I, float *D, int w, int h, int nc,
		float tau)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z + blockDim.z * blockIdx.z;

	float upd;

	if (x < w && y < h)
	{
		float i = read_data(I, w, h, nc, x, y, c);
		float d = read_data(D, w, h, nc, x, y, c);
		upd = i + tau * d;
		write_data(I, upd, w, h, nc, x, y, c);
	}
}
