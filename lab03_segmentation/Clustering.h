#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

void OtsuTreshold(cv::Mat& source);

void CannyBorderDetection(cv::Mat& source);

void Watershed(cv::Mat& source);

