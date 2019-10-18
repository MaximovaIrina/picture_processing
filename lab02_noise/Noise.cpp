#include "Noise.h"
#include "boost/math/special_functions/lambert_w.hpp"

using boost::math::lambert_w0;

double Fact(int n) {
  if (n < 0) return 0;
  if (n == 0)
    return 1;
  else
    return n * Fact(n - 1);
}

void GammaNoise(Mat& image, const vector<float> a, const vector<int> b) {
  Mat noise(image.size(), image.type());
  RNG rng;
  double r, res, z, l;

  for (int i = 0; i < noise.rows; ++i)
    for (int j = 0; j < noise.cols; ++j)
      if (rng.uniform(0, 100) >= 50)
        for (int k = 0; k < 3; ++k) {
          r = rng.uniform((double)0, (double)1);
          z = -a[k] *
              powf(pow(a[k], -b[k]) * r * Fact(b[k] - 1),
                   1. / (float)(b[k] - 1)) /
              (float)(b[k] - 1);
          l = 0;
          if (z >= -0.367879) l = lambert_w0(z);
          res = -((b[k] - 1) / a[k]) * l;
          res = res * 30;
          noise.at<Vec3b>(i, j)[k] = res;
        }
      else
        noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
  GetHistRGB(noise);

  image += noise;
  imwrite("GammaNoise.jpg", image);
}

void GaussianNoise(Mat& image, const vector<int> m, const vector<float> dis) {
  Mat noise(image.size(), image.type());
  RNG rng;
  double r, res;

  for (int i = 0; i < noise.rows; i++)
    for (int j = 0; j < noise.cols; j++)
      if (rng.uniform(0, 100) >= 50)
        for (int k = 0; k < 3; ++k) {
          r = rng.uniform((double)0, (double)1);
          res = m[k] + sqrt(2 * pow(dis[k], 2) * log(1. / (r * dis[k] * 2.5)));
          noise.at<Vec3b>(i, j)[k] = res;
        }
      else
        noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
  GetHistRGB(noise);

  image += noise;
  imwrite("GaussianNoise.jpg", image);
}

void ExponentialNoise(Mat& image, const vector<float> a) {
  Mat noise(image.size(), image.type());
  RNG rng;
  double res, r;

  for (int i = 0; i < noise.rows; ++i)
    for (int j = 0; j < noise.cols; ++j)
      if (rng.uniform(0, 100) >= 50)
        for (int k = 0; k < 3; ++k) {
          r = rng.uniform((double)0, (double)1);
          res = log(1. / (1 - r)) / a[k];
          noise.at<Vec3b>(i, j)[k] = res;
        }
      else
        noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
  GetHistRGB(noise);

  image += noise;
  imwrite("ExponentialNoise.jpg", image);
}

void RayleightNoise(Mat& image, vector<float> a, vector<float> disp) {
  Mat noise(image.size(), image.type());
  RNG rng;
  double r, res, l, z;

  for (int i = 0; i < noise.rows; i++)
    for (int j = 0; j < noise.cols; j++)
      if (rng.uniform(0, 100) >= 50)
        for (int k = 0; k < 3; ++k) {
          r = rng.uniform((double)0, (double)1);
          res = a[k] + sqrt(2 * powf(disp[k], 2) * log(1. / (1 - r)));
          noise.at<Vec3b>(i, j)[k] = res;
        }
      else
        noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
  GetHistRGB(noise);

  image += noise;
  imwrite("RayleightNoise.jpg", image);
}

void ConstantNoise(Mat& image, vector<int> a, vector<int> b) {
  Mat noise(image.size(), image.type());
  RNG rng;
  int r;

  for (int i = 0; i < noise.rows; i++)
    for (int j = 0; j < noise.cols; j++)
      if (rng.uniform(0, 100) >= 50)
        for (int k = 0; k < 3; ++k) {
          r = rng.uniform(a[k], b[k]);
          noise.at<Vec3b>(i, j)[k] = r;
        }
      else
        noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
  GetHistRGB(noise);

  image += noise;
  imwrite("ConstantNoise.jpg", image);
}

void SaltPepperNoise(Mat& image, vector<int> min, vector<int> max) {
  Mat noise(image.size(), image.type());
  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++) noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
  RNG rng;
  int r;

  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++)
      if (rng.uniform(0, 100) >= 50)
        for (int k = 0; k < 3; ++k) {
          r = rng.uniform(0, 256);
          if (r < min[k]) noise.at<Vec3b>(i, j)[k] = min[k];
          if (r > max[k]) noise.at<Vec3b>(i, j)[k] = max[k];
        }
  GetHistRGB(noise);

  image += noise;
  imwrite("SaltPepperNoise.jpg", image);
}
