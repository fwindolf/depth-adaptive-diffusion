#ifndef GRADIENT_H
#define GRADIENT_H

/**
 * Compute the l2 norm between channels of a gradient in two images
 * Return a greyscale image
 */
__global__ void g_l2norm(float * V_1, float * V_2, float *O, int w, int h, int nc);

/**
 * Compute the l2 norm between channels and return a greyscale image
 */
__global__ void g_l2norm(float * I, float *O, int w, int h, int nc);

/**
 * Compute the gradient with forward difference
 */
__global__ void g_gradient(float *I, float *V_1, float *V_2, int w, int h, int nc);

/**
 * Compute the divergence of two gradients
 */
__global__ void g_divergence(float * V_1, float * V_2, float *D, int w, int h, int nc);




#endif /* GRADIENT_H */
