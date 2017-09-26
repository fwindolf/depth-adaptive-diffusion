#include "image.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

#define MAX_W 1024
#define MAX_H 768

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

void convert_layered_to_mat(Mat &mOut, const float *aIn)
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

void convert_mat_to_layered(float *aOut, const Mat &mIn)
{
	convert_interleaved_to_layered(aOut, (float*) mIn.data, mIn.cols, mIn.rows,
			mIn.channels());
}

void showImage(string title, const Mat &mat, int x, int y)
{
	const char *wTitle = title.c_str();
	namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(wTitle, x, y);
	imshow(wTitle, mat);
}

void showHistogram256(const char *windowTitle, int *histogram, int windowX,
		int windowY)
{
	const int nbins = 256;
	Mat canvas = Mat::ones(125, 512, CV_8UC3);

	float hmax = 0;
	for (int i = 0; i < nbins; ++i)
		hmax = max((int) hmax, histogram[i]);

	for (int j = 0, rows = canvas.rows; j < nbins - 1; j++)
	{
		for (int i = 0; i < 2; ++i)
			line(canvas, Point(j * 2 + i, rows),
					Point(j * 2 + i, rows - (histogram[j] * 125.0f) / hmax),
					Scalar(255, 128, 0), 1, 8, 0);
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

void addNoise(Mat &m, float sigma)
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

void downsample(Mat &mIn, int max_w, int max_h)
{
	float factor = 0.f;
	// downsample if bigger than MAX_W or MAX_H
	if (mIn.cols > max_w)
		factor = (float) max_w / mIn.cols;

	if (mIn.rows > MAX_H)
		factor = min((float) max_h / mIn.rows, factor);

	if (factor > 0)
		resize(mIn, mIn, Size(), factor, factor, INTER_AREA);
}


Mat load_image(const std::string image, bool gray)
{
	// Load the input image using opencv
	// (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
	Mat mIn = imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
	// check
	if (mIn.data == NULL)
	{
		cerr << "ERROR: Could not load image " << image << endl;
		exit(1);
	}

	mIn.convertTo(mIn, CV_32F);
	mIn /= 255.f;

	// cout << "Original image is " << mIn.cols << " x " << mIn.rows << " x " << mIn.channels() << endl;

	downsample(mIn, MAX_W, MAX_H);

	return mIn;
}


/**
 * Implementation reference:
 * https://github.com/antoinetlc/PFM_ReadWrite/blob/master/PFMReadWrite.cpp
 */
Mat load_pfm(const std::string image)
{
	// Open image as binary filestream
	ifstream file(image.c_str(), ios::in | ios::binary);

	Mat mDisparities;

	// check if opened correctly
	if (file)
	{
		// filetype plus 0x0a Unix return
		char type[3];
		file.read(type, 3 * sizeof(char));

		// width and height
		unsigned int width = 0, height = 0;
		file >> width >> height;

		// 0x0a Unix return
		char eol;
		file.read(&eol, sizeof(char));

		int channels = 0;
		// type[1] indicates number of channels
		if (type[1] == 'F')
		{
			mDisparities = Mat(height, width, CV_32FC3);
			channels = 3;
		}
		else if (type[1] == 'f')
		{
			mDisparities = Mat(height, width, CV_32FC1);
			channels = 1;
		}

		// endianess plus 0x0a Unix return
		char byteorder[4];
		file.read(byteorder, 4 * sizeof(char));

		// read until pixels start
		char returnchar = ' ';
		while (returnchar != 0x0a)
		{
			file.read(&returnchar, sizeof(char));
		}

		// read all pixel to matrix
		float *color = new float[channels];
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				// next pixel, values from 0-255 (char)
				file.read((char *) color, channels * sizeof(float));
				if (channels == 3)
				{
					// OpenCV stores color as BGR, pfm is RGB
					mDisparities.at<Vec3f>((height - 1) - y, x) = Vec3f(
							color[2], color[1], color[0]);
				}
				else if (channels == 1)
				{
					mDisparities.at<float>((height - 1) - y, x) = color[0];
				}
			}
		}

		// tidy up
		delete[] color;
	}
	else
	{
		cerr << "Unable ot open file " << image << endl;
		exit(1);
	}

	// close filestream
	file.close();

	// Change range (0-255) to (0-1)
	mDisparities /= 255.f;

	// downsample to max resolution
	downsample(mDisparities, MAX_W, MAX_H);

	return mDisparities;
}

void get_dimensions(const Mat &m1, const Mat &m2, int &w, int &h, int &nc)
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

void get_dimensions(const Mat &m, int &w, int &h, int &nc)
{
	w = m.cols;        // width
	h = m.rows;        // height
	nc = m.channels(); // number of channels
}


void save_image(string image_name, Mat &mOut)
{
	// save input and result
	imwrite(image_name + ".png", mOut * 255.f);
}

