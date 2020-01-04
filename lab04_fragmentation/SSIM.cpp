#include "SSIM.h"
#include <math.h>
#include <vector>

float SSIM(Mat& x, Mat& y) {
  double mx = 0, my = 0, sigmax = 0, sigmay = 0, cov = 0;
  int size = x.rows * x.cols;

  for (int i = 0; i < x.rows; ++i)
    for (int j = 0; j < x.cols; ++j) {
      mx += x.at<uchar>(i, j);
      my += y.at<uchar>(i, j);
    }

  mx /= size;
  my /= size;

  for (int i = 0; i < x.rows; ++i)
    for (int j = 0; j < x.cols; ++j) {
      sigmax += powf(x.at<uchar>(i, j) - mx, 2);
      sigmay += powf(y.at<uchar>(i, j) - my, 2);
      cov += (x.at<uchar>(i, j) - mx) * (y.at<uchar>(i, j) - my);
    }

  sigmax /= (size - 1);
  sigmay /= (size - 1);
  cov /= size;

  float p = 0.01, L, C, S;
  L = (2 * mx * my + p) / (powf(mx, 2) + powf(my, 2) + p);
  C = (2 * sqrt(sigmax * sigmay) + p) / (sigmax + sigmay + p);
  S = (cov + p) / (sqrt(sigmax * sigmay) + p);

  float res = L * C * S;
  return res;
}
