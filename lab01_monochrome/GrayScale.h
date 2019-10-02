#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void Average(Mat* im);
void Lightness(Mat* im);
void Luminosity(Mat* im);
void GIMP(Mat* im);
void UTUR(Mat* im);
void MaxRGB(Mat* im);
void MinRGB(Mat* im);
void Cust(Mat* im);
