#ifndef KERNELS_H
#define KERNELS_H

#include "kernels.h"
#include "util.h"

/**
 * Compute a global rho
 *
 * x * y threads needed
 *
 * Rho is the error values for different disparities
 * IL and IR are the left/right original images with x * y * nc
 */
__global__ void g_compute_rho(float *iL, float *iR, float *Rho, int w, int h,
		int nc, int gamma_min, int gamma_max, float lambda);

/**
 * Initialize Phi with entries 1.0 in layer gamma_min
 *
 * x * y threads needed
 */
__global__ void g_init_phi(float *Phi, int w, int h, int gc);

/**
 * Update and backproject the Phi vector and save the result back to Phi
 *
 * x * y threads needed
 *
 * Phi is dimensions (w * h * gc) and can be in [0, 1], thus the value gets
 * clamped back into the interval in the projection
 */
__global__ void g_update_phi(float *Phi, float *Div3_P, int w, int h, int gc,
		float tau_p);

/**
 * Update and backproject the P vector and save the result back to P
 *
 * x * y threads needed
 *
 * P = (p1, p2, p3) with dimensions (w * h * gc * 3)
 * Grad3_Phi is the gradient of Phi in x, y and gamma direction (w * h * gc * 3)
 */
__global__ void g_update_p(float *P, float *Grad3_Phi, float *Rho, int w, int h,
		int gc, float tau_d);

/**
 * Calculate the gradient in x, y and gamma direction
 * Phi is dimensions (w * h * gc)
 *
 * x * y threads needed
 *
 * Grad3_Phi is the resulting w * h * gc * 3 with one channel for x, y and g direction
 */
__global__ void g_grad3(float *Phi, float *Grad3_Phi, int w, int h, int gc,
		float dx, float dy, float dg);

/**
 * Calculate the divergence of P
 *
 * x * y threads needed
 *
 * P is stored in (w * h * gc * 3), so (p1, p2, p3)
 * Div3_P has the same dimensions as Phi (w * h * gc)
 */
__global__ void g_div3(float *P, float *Div3_P, int w, int h, int gc, float dx,
		float dy, float dg);

/**
 * Calculate the function u(x) (=G) from Phi
 *
 * x * y threads neeeded
 */
__global__ void g_compute_u(float *Phi, float *U, int w, int h, int gamma_min,
		int gamma_max);

/**
 * Calcualate the energy of the current gamma matrix
 *
 * x * y threads needed
 */
__global__ void g_compute_energy(float * U, float *IL, float *IR,
		float * energy, int w, int h, int nc, float lambda);

/**
 * Calculate the depth from the disparity values
 *
 * Z = baseline * f / (d + doffs)
 *
 * baseline is the camera baseline in mm
 * f is the focal length in pixels
 * doffs is the x-difference of principal points (cx1 - cx0) for im1 and im0
 *
 * The result will be stored in Depths
 */
__global__ void g_compute_depth(float * Disparities, float *Depths, int w,
		int h, float baseline, int f, int doffs);

/**
 * Compute the g matrix from the Depths
 *
 * x * y threads needed
 *
 * Depths hold the depth values normalized to [0,1] for the image
 * z_f is the point of focus between [0,1]
 * radius indicates how much the points not in focus will be diffused
 *
 * The result will be stored in G
 */
__global__ void g_compute_g_matrix(float *Depths, float *G, int w, int h,
		float z_f, float radius);

/**
 * Apply the G matrix to the gradients
 *
 * x * y threads needed
 *
 * Grad_x and Grad_y hold the gradients of the image
 * G is the G matrix
 *
 * The result will be stored back in the gradients
 */
__global__ void g_apply_g(float *Grad_x, float *Grad_y, float *G, int w, int h,
		int nc);

/**
 * Update step for the image during the diffusion
 *
 * x * y * nc threads needed
 *
 * I is the image of the step
 * D is the result of the divergence of the last step
 * tau is the step size
 *
 * The output will be stored back into I
 */
__global__ void g_update_step(float *I, float *D, int w, int h, int nc,
		float tau);

#endif
