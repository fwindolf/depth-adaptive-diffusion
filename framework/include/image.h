#ifndef IMAGE_H
#define IMAGE_H

#include "util.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


/**
 * Convert a matrix in interleaved format to layered format
 */
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn);

/**
 * Convert an image in layered format to interleaved matrix
 */
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn);

/**
 * Display an image in the frame with title at x, y
 */
void showImage(std::string title, const cv::Mat &mat, int x, int y);

/**
 * Display a histogram at x, y
 */
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY);


/**
 * Add gaussian noise with intensity sigma to an image
 */
void addNoise(cv::Mat &m, float sigma);

/**
 * Read an input image
 */
cv::Mat load_image(std::string image, bool gray);

/**
 * Save the dimensions of the images to w, h, and nc
 *
 * This will exit the program if the dimensions of m1 and m2 dont match
 */
void get_dimensions(const cv::Mat &m1, const cv::Mat &m2, int &w, int &h,
		int &nc);

#endif //IMAGE_H




