#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

std::vector<float> SSIM(Mat& x, Mat& y);

void GetImage(Mat& x, Mat& y);