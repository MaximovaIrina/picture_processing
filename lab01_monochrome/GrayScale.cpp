#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include "GrayScale.h"

using namespace cv;
using namespace std;

void Average(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = (im->at<Vec3b>(i, j)[0] + im->at<Vec3b>(i, j)[1] + im->at<Vec3b>(i, j)[2]) / 3;
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void Lightness(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			int rgbmax = MAX(im->at<Vec3b>(i, j)[0], im->at<Vec3b>(i, j)[1], im->at<Vec3b>(i, j)[2]);
			int rgbmin = MIN(im->at<Vec3b>(i, j)[0], im->at<Vec3b>(i, j)[1], im->at<Vec3b>(i, j)[2]);
			im->at<Vec3b>(i, j)[0] = (rgbmax + rgbmin) / 2;
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void Luminosity(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = 0.21 * im->at<Vec3b>(i, j)[0] + 0.72 * im->at<Vec3b>(i, j)[1] + 0.07 * im->at<Vec3b>(i, j)[2];
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void GIMP(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = 0.3 * im->at<Vec3b>(i, j)[0] + 0.59 * im->at<Vec3b>(i, j)[1] + 0.11 * im->at<Vec3b>(i, j)[2];
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void UTUR(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = 0.2126 * im->at<Vec3b>(i, j)[0] + 0.7152 * im->at<Vec3b>(i, j)[1] + 0.0722 * im->at<Vec3b>(i, j)[2];
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void MaxRGB(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = MAX(im->at<Vec3b>(i, j)[0], im->at<Vec3b>(i, j)[1], im->at<Vec3b>(i, j)[2]);
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void MinRGB(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = MIN(im->at<Vec3b>(i, j)[0], im->at<Vec3b>(i, j)[1], im->at<Vec3b>(i, j)[2]);
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}

void Cust(Mat* im)
{
	for (int i = 0; i < im->rows; i++)
		for (int j = 0; j < im->cols; j++)
		{
			im->at<Vec3b>(i, j)[0] = 0.2952 * im->at<Vec3b>(i, j)[0] + 0.148 * im->at<Vec3b>(i, j)[1] + 0.0722 * im->at<Vec3b>(i, j)[2];
			im->at<Vec3b>(i, j)[1] = im->at<Vec3b>(i, j)[0];
			im->at<Vec3b>(i, j)[2] = im->at<Vec3b>(i, j)[0];
		}
}