#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Fragmentation.h"
#include "Processing.h"
#include "SSIM.h"

using namespace std;

void GaussFilterPr(Mat& image, Mat& result, int radius, double sigma) {
  double** kernel = new double*[radius];
  for (int i = 0; i < radius; i++) kernel[i] = new double[radius];

  if (sigma <= 0.000000001) sigma = 1.398;

  double sum = 0;

  for (int x = -radius / 2; x <= radius / 2; x++) {
    for (int y = -radius / 2; y <= radius / 2; y++) {
      kernel[x + radius / 2][y + radius / 2] =
          (exp((x * x + y * y) / (2 * sigma * sigma * (-1.0)))) /
          (3.14159265359 * 2 * sigma * sigma);
      sum += kernel[x + radius / 2][y + radius / 2];
    }
  }

  for (int i = 0; i < radius; i++)
    for (int j = 0; j < radius; j++)
      kernel[i][j] = kernel[i][j] / sum;  // нормализация

  uchar color;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      color = 0;
      for (int p = 0; p < radius; p++) {
        for (int q = 0; q < radius; q++) {
          int tmp1 = i + p;
          while (tmp1 >= image.rows) tmp1--;
          int tmp2 = j + q;
          while (tmp2 >= image.cols) tmp2--;
          color += kernel[p][q] * image.at<uchar>(tmp1, tmp2);
        }
      }
      result.at<uchar>(i, j) = color;
    }
  }

  for (int i = 0; i < radius; i++) delete[] kernel[i];
  delete[] kernel;
}

int main() {
  // Выбор операции
  int var = 0;
  cout << "\n\n1. LowPassFilter\n2. HightPassFilter\n";
  cout << "3. Gauss Filter\n4. PeriodicNoiseRemoval\n";
  while (var > 4 || var < 1) {
    cout << "\nEnter operation number: ";
    cin >> var;
  }

  // Только квадратные изображения 192x192 !!!
  Mat source, ideal;
  ideal = imread("ideal.jpg");
  switch (var) {
    case 1: {
      source = imread("noise.jpg");
      break;
    }
    case 2: {
      source = imread("ideal.jpg");
      break;
    }
    case 3: {
      source = imread("noise.jpg");
      break;
    }
    case 4: {
      source = imread("periodicNoise.jpg");
      break;
    }
    default:
      break;
  }

  cvtColor(ideal, ideal, COLOR_BGR2GRAY);
  cvtColor(source, source, COLOR_BGR2GRAY);

  Mat fourier = source.clone();
  imshow("Ideal", ideal);
  waitKey(0);
  imshow("Source", source);
  waitKey(0);

  // Преобразоание Фурье
  int n = source.rows;
  pair<float, float>** Fuv = new pair<float, float>*[n];
  for (int i = 0; i < n; ++i) Fuv[i] = new pair<float, float>[n];
  FourierTransform(source, fourier, Fuv);
  imshow("Fourier_Transform", fourier);
  waitKey(0);

  pair<float, float>** FuvRes = new pair<float, float>*[n];
  for (int i = 0; i < n; ++i) FuvRes[i] = new pair<float, float>[n];

  Mat result = source.clone();
  Mat filter = source.clone();

  switch (var) {
    case 1: {
      LowPassFilter(Fuv, FuvRes, n, 60, filter);
      InverseFourierTransform(result, FuvRes);
      cout << "SSIM = " << SSIM(ideal, result) << endl;
      break;
    }
    case 2: {
      HightPassFilter(Fuv, FuvRes, n, 20, filter);
      InverseFourierTransform(result, FuvRes);
      break;
    }
    case 3: {
      // radFilter - нечет
      Gauss_Fourier_Filter(Fuv, FuvRes, n, 5, 1, filter);

      // Обратное преобразование Фурье
      Mat temp = source.Mat::clone();
      InverseFourierTransform(temp, FuvRes);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          result.at<uchar>(i, j) =
              temp.at<uchar>(((i + n / 2) % n), ((j + n / 2) % n));

      Mat resultPr = source.clone();
      GaussFilterPr(source, resultPr, 5, 1);
      imshow("Pr", resultPr);
      waitKey(0);
      cout << "SSIM = " << SSIM(ideal, resultPr) << endl;
      cout << "SSIM = " << SSIM(ideal, result) << endl;
      break;
    }
    case 4: {
      PeriodicNoiseRemoval(Fuv, FuvRes, n, 20, 65, filter);
      InverseFourierTransform(result, FuvRes);
      break;
    }
    default:
      break;
  }
  imshow("Filter", filter);
  waitKey(0);

  imshow("Inverse_Fourier_Transform", result);
  waitKey(0);

  return 0;
}