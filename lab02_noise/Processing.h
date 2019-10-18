#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void Binarization(cv::Mat& image, cv::Mat& result);

void MedianFilter(cv::Mat& image, cv::Mat& result, int n);

void MeanFilter(cv::Mat& image, cv::Mat& result);

void GaussFilter(cv::Mat& image, cv::Mat& result, int radius, double sigma);

void MorphOpening(cv::Mat& image, cv::Mat& result);

void MorphClosing(cv::Mat& image, cv::Mat& result);

void MidpointFilter(cv::Mat& image, cv::Mat& result, int size);
