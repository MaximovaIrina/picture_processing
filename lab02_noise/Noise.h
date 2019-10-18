#include "Histogram.h"

void GammaNoise(Mat& image, const vector<float> a, const vector<int> b);

void GaussianNoise(Mat& image, const vector<int> m, const vector<float> dis);

void ExponentialNoise(Mat& image, const vector<float> a);

void RayleightNoise(Mat& image, vector<float> a, vector<float> b);

void ConstantNoise(Mat& image, vector<int> a, vector<int> b);

void SaltPepperNoise(Mat& image, vector<int> min, vector<int> max);