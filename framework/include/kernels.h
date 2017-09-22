#ifndef KERNELS_H
#define KERNELS_H

#include "kernels.h"
#include "util.h"

/**
 * Rho function implementation
 *
 * Calculates the absolute error for each pixel
 *
 * iL is the pixel values at x, y of the left image for all channels nc
 * iR is the pixel values at x  plus a current disparity of the right image for all nc
 * lambda is some constant parameter
 */
__device__ float rho(float *iL, float *iR, int nc, float lambda);

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
		int nc, int gc, float lambda, float gamma_min);

/**
 * Project the Phi vector back onto D
 *
 * x * y threads needed
 *
 * Phi is dimensions (w * h * gc) and can be in [0, 1], thus the value gets
 * clamped back into the interval in the projection
 */
__global__ void g_project_phi_d(float * Phi, int w, int h, int gc);
/**
 * Update the P vector and save the result back to P
 *
 * x * y threads needed
 *
 * P = (p1, p2, p3) with dimensions (w * h * gc * 3)
 * Grad3_Phi is the gradient of Phi in x, y and gamma direction (w * h * gc * 3)
 */
__global__ void g_update_p(float * P, float *Grad3_Phi, int w, int h, int gc,
		float tau_d);

/**
 * Update the Phi vector and save the result back to Phi
 *
 * x * y threads needed
 *
 * Phi and Div3_P are dimensions (w * h * gc)
 */
__global__ void g_update_phi(float *Phi, float *Div3_P, int w, int h, int gc,
		float tau_p);

/**
 * Calculate the gradient in x, y and gamma direction
 * Phi is dimensions (w * h * gc)
 *
 * x * y threads needed
 *
 * Grad3_Phi is the resulting w * h * gc * 3 with one channel for x, y and g direction
 */
__global__ void g_grad3(float *Phi, float *Grad3_Phi, int w, int h, int gc);

/**
 * Calculate the divergence of P
 *
 * x * y threads needed
 *
 * P is stored in (w * h * gc * 3), so (p1, p2, p3)
 * Div3_P has the same dimensions as Phi (w * h * gc)
 */
__global__ void g_div3(float *P, float *Div3_P, int w, int h, int gc);

/**
 * Calculate the function u(x) (=G) from Phi
 *
 * x * y threads neeeded
 */
__global__ void g_compute_g(float *Phi, float *G, int w, int h, int gamma_min,
		int gamma_max);

#endif
