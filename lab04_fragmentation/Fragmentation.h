#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <utility>
using namespace cv;
using namespace std;

void FourierTransform(Mat& source, Mat& fourier, pair<float, float>** Fuv);
void InverseFourierTransform(Mat& result, pair<float, float>** Fuv);
void LowPassFilter(pair<float, float>** Fuv, pair<float, float>** FuvRes, int n,
                   int rad, Mat& filter);
void HightPassFilter(pair<float, float>** Fuv, pair<float, float>** FuvRes,
                     int n, int rad, Mat& filter);
void Gauss_Fourier_Filter(pair<float, float>** Fuv, pair<float, float>** FuvRes,
                          int n, int radFilter, double sigma, Mat& filter);
void PeriodicNoiseRemoval(pair<float, float>** Fuv, pair<float, float>** FuvRes,
                          int n, int gr, int size_kernel, Mat& filter);

