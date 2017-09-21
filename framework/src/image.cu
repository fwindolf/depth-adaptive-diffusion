#include "image.h"
#include <string>
#include <iostream>

using namespace std;

#define MAX_W 800
#define MAX_H 600

// opencv helpers
void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h,
		int nc)
{
	if (nc == 1)
	{
		memcpy(aOut, aIn, w * h * sizeof(float));
		return;
	}
	size_t nOmega = (size_t) w * h;
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			for (int c = 0; c < nc; c++)
			{
				aOut[(nc - 1 - c) + nc * (x + (size_t) w * y)] = aIn[x
						+ (size_t) w * y + nOmega * c];
			}
		}
	}
}
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn)
{
	convert_layered_to_interleaved((float*) mOut.data, aIn, mOut.cols,
			mOut.rows, mOut.channels());
}

void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h,
		int nc)
{
	if (nc == 1)
	{
		memcpy(aOut, aIn, w * h * sizeof(float));
		return;
	}
	size_t nOmega = (size_t) w * h;
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			for (int c = 0; c < nc; c++)
			{
				aOut[x + (size_t) w * y + nOmega * c] = aIn[(nc - 1 - c)
						+ nc * (x + (size_t) w * y)];
			}
		}
	}
}
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn)
{
	convert_interleaved_to_layered(aOut, (float*) mIn.data, mIn.cols, mIn.rows,
			mIn.channels());
}

void showImage(string title, const cv::Mat &mat, int x, int y)
{
	const char *wTitle = title.c_str();
	cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(wTitle, x, y);
	cv::imshow(wTitle, mat);
}

void showHistogram256(const char *windowTitle, int *histogram, int windowX,
		int windowY)
{
	const int nbins = 256;
	cv::Mat canvas = cv::Mat::ones(125, 512, CV_8UC3);

	float hmax = 0;
	for (int i = 0; i < nbins; ++i)
		hmax = max((int) hmax, histogram[i]);

	for (int j = 0, rows = canvas.rows; j < nbins - 1; j++)
	{
		for (int i = 0; i < 2; ++i)
			cv::line(canvas, cv::Point(j * 2 + i, rows),
					cv::Point(j * 2 + i, rows - (histogram[j] * 125.0f) / hmax),
					cv::Scalar(255, 128, 0), 1, 8, 0);
	}

	showImage(windowTitle, canvas, windowX, windowY);
}

// adding Gaussian noise
float noise(float sigma)
{
	float x1 = (float) rand() / RAND_MAX;
	float x2 = (float) rand() / RAND_MAX;
	return sigma * sqrtf(-2 * log(std::max(x1, 0.000001f)))
			* cosf(2 * M_PI * x2);
}

void addNoise(cv::Mat &m, float sigma)
{
	float *data = (float*) m.data;
	int w = m.cols;
	int h = m.rows;
	int nc = m.channels();
	size_t n = (size_t) w * h * nc;
	for (size_t i = 0; i < n; i++)
	{
		data[i] += noise(sigma);
	}
}

cv::Mat load_image(std::string image, bool gray)
{
	// Load the input image using opencv
	// (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
	cv::Mat mIn = cv::imread(image.c_str(),
			(gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
	// check
	if (mIn.data == NULL)
	{
		cerr << "ERROR: Could not load image " << image << endl;
		exit(1);
	}

	mIn.convertTo(mIn, CV_32F);
	mIn /= 255.f;

	// cout << "Original image is " << mIn.cols << " x " << mIn.rows << " x " << mIn.channels() << endl;

	float factor = 0.f;
	// downsample if bigger than MAX_W or MAX_H
	if(mIn.cols > MAX_W)
		factor = (float)MAX_W / mIn.cols;

	if(mIn.rows > MAX_H)
		factor = min((float)MAX_H / mIn.rows, factor);

	if(factor > 0)
		cv::resize(mIn, mIn, cv::Size(), factor, factor, cv::INTER_AREA);

	return mIn;
}

void get_dimensions(const cv::Mat &m1, const cv::Mat &m2, int &w, int &h,
		int &nc)
{
	w = m1.cols;        // width
	h = m1.rows;        // height
	nc = m1.channels(); // number of channels

	// check if dimensions match
	if (w != m2.cols || h != m2.rows || nc != m2.channels())
	{
		cerr << "ERROR: image dimensions mismatch!" << endl;
		exit(1);
	}

}

