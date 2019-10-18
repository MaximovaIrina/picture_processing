#include <iostream>
#include <vector>
#include "Noise.h"
#include "NoiseMono.h"
#include "Processing.h"
#include "SSIM.h"

using namespace cv;

int main(int argc, char* argv[]) {
  Mat source = imread("adel.jpg");
  Mat noise = source.clone();
  Mat filter = source.clone();
  // Mat bin = source.clone();
  //Mat filterSTD = source.clone();
  // cvtColor(source, noise, COLOR_BGR2GRAY);

  //  Binarization(source, image);

  // ************************ //
  // *** Generating Noise *** //
  // ************************ //

  // ****** RGB Noise ******* //

  // ***** Gamma Noise ****** //
  // vector<float> a = {4.5, 4.5, 4.5};
  // vector<int> b = {4, 5, 8};
  // GammaNoise(noise, a, b);

  // **** Gaussian Noise **** //
  // vector<int> m = {140, 120, 100};
  // vector<float> dis = {2, 1, 0.01};
  // GaussianNoise(noise, m, dis);

  // *** Exponential Noise *** //
  // vector<float> a = {0.05, 0.1, 1};
  // ExponentialNoise(noise, a);

  // **** Rayleight Noise *** //
  // vector<float> a = {10, 10, 10};
  // vector<float> disp = {2, 10, 20};
  // RayleightNoise(noise, a , disp);

  // **** Constant Noise **** //
  // vector<int> a = {30, 50, 80};
  // vector<int> b = {35, 60, 100};
  // ConstantNoise(noise, a, b);

  // **** SaltPepper Noise **** //
  // vector<int> min = {10, 20, 30};
  // vector<int> max = {210, 220, 230};
  // SaltPepperNoise(noise, min, max);

  namedWindow("Noise", WINDOW_AUTOSIZE);
  imshow("Noise", noise);

  // ************************ //
  // **** Filtering Noise *** //
  // ************************ //

  // MedianFilter(noise, filter, 1);
  // MeanFilter(noise, filter);
  // GaussFilter(noise, filter, 3, 4);
  // MidpointFilter(noise, filter, 5);

  // MorphOpening(noise, filter);
  // MorphClosing(noise, filter);

  // ************************ //
  // ** Standart Filtering ** //
  // ************************ //

  // GaussianBlur(noise, filterSTD, Size(3, 3), 0, 0);
  // medianBlur(noise, filterSTD, 5);

  namedWindow("Filter", WINDOW_AUTOSIZE);
  imshow("Filter", filter);

  std::vector<float> ssim;
  ssim = SSIM(source, filter);
  // ssim = SSIM(filter, filterSTD);
  
  float res = (ssim[0] + ssim[1] + ssim[2]) / 3;
  std::cout << res << std::endl;
  
  waitKey();
  return 0;
}