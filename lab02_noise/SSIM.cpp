#include "SSIM.h"
#include <math.h>
#include <vector>

std::vector<float> SSIM(Mat& x, Mat& y) {
  double mx[3] = {}, my[3] = {}, sigmax[3] = {}, sigmay[3] = {}, cov[3] = {};
  int size = x.rows * x.cols;

  for (int i = 0; i < x.rows; ++i)
    for (int j = 0; j < x.cols; ++j)
      for (int k = 0; k < 3; ++k) {
        mx[k] += x.at<Vec3b>(i, j)[k];
        my[k] += y.at<Vec3b>(i, j)[k];
      }

  for (int k = 0; k < 3; ++k) {
    mx[k] = mx[k] / size;
    my[k] = my[k] / size;
  }

  for (int i = 0; i < x.rows; ++i)
    for (int j = 0; j < x.cols; ++j)
      for (int k = 0; k < 3; ++k) {
        sigmax[k] += powf(x.at<Vec3b>(i, j)[k] - mx[k], 2);
        sigmay[k] += powf(y.at<Vec3b>(i, j)[k] - my[k], 2);
        cov[k] +=
            (x.at<Vec3b>(i, j)[k] - mx[k]) * (y.at<Vec3b>(i, j)[k] - my[k]);
      }

  for (int k = 0; k < 3; ++k) {
    sigmax[k] = sigmax[k] / (size - 1);
    sigmay[k] = sigmay[k] / (size - 1);
    cov[k] = cov[k] / size;
  }

  float p = 0.01, L[3] = {}, C[3] = {}, S[3] = {};
  for (int k = 0; k < 3; ++k) {
    L[k] = (2 * mx[k] * my[k] + p) / (powf(mx[k], 2) + powf(my[k], 2) + p);
    C[k] = (2 * sqrt(sigmax[k] * sigmay[k]) + p) / (sigmax[k] + sigmay[k] + p);
    S[k] = (cov[k] + p) / (sqrt(sigmax[k] * sigmay[k]) + p);
  }

  std::vector<float> res;
  for (int k = 0; k < 3; ++k) res.push_back(L[k] * C[k] * S[k]);
  return res;
}

void GetImage(Mat& x, Mat& y) {
  for (int i = 0; i < x.rows; ++i) {
    for (int j = 0; j < x.cols; ++j)
      for (int k = 0; k < 3; ++k)
        std::cout << x.at<Vec3b>(i, j)[k] - y.at<Vec3b>(i, j)[k] << " ";
    std::cout << std::endl;
  }
}