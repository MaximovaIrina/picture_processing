#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "GrayScale.h"
#include "math.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	String imageName("picture.jpg"); // by default
	if (argc > 1)
		imageName = argv[1];

	Mat imagesrc = imread(imageName, IMREAD_COLOR);
	Mat image(imagesrc);

	if (imagesrc.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//Average(&image);
	//Lightness(&image);
	//Luminosity(&image);
	//GIMP(&image);
	//UTUR(&image);
	//MaxRGB(&image);
	//MinRGB(&image);
	//Cust(&image);
	//cvtColor(image, image, COLOR_BGR2GRAY);

	double sum_sq = 0, mse, pix1, pix2, err;

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
		{
			pix1 = imagesrc.at<uchar>(i, j);
			pix2 = image.at<uchar>(i, j);
			err = pix2 - pix1;
			sum_sq += pow(err, 2);
		}

	mse = sum_sq / (image.rows * image.cols);
	cout << "MSE = " << mse << endl;

	double psnr = 10 * log(255 * 255 / mse);
	cout << "PSNR = " << psnr << endl;

	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);
	waitKey(0);
	return 0;
}
