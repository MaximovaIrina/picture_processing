#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Clustering.h"

using namespace cv;
using namespace std;

int main() {
  Mat image;
  image = imread("money.jpg");
  imshow("Source", image);
  waitKey(0);

  OtsuTreshold(image);
  Watershed(image);

  CannyBorderDetection(image);
  return 0;
}