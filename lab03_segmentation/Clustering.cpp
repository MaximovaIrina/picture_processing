#include "Clustering.h"
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <stack>
#include <vector>

#define M_PI 3.14159265358979323846
#define ErrCode -1

struct regions {
  int n;
  int m;
  int count;
  int** arr;
  std::vector<float> regionSize;          //размер региона
  std::vector<float> regionSumIntensity;  //размер региона

  regions(int _n, int _m) {
    count = 0;
    n = _n;
    m = _m;
    arr = new int*[n];
    for (int i = 0; i < n; ++i) arr[i] = new int[m];

    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j) arr[i][j] = 0;

    regionSize.push_back(0);
    regionSumIntensity.push_back(0);
  }

  ~regions() {
    for (int i = 0; i < n; ++i) delete[] arr[i];
    delete[] arr;
  }

  void createRegion(int intensity, int x, int y) {
    count++;
    arr[x][y] = count;
    regionSize.push_back(1.);
    regionSumIntensity.push_back(intensity);
  }

  void addElem(int intensity, int x, int y, int region) {
    arr[x][y] = region;
    regionSize[region] += 1;
    regionSumIntensity[region] += intensity;
  }

  double clavg(int region) {
    if (region) return (regionSumIntensity[region] / regionSize[region]);
    return 0;
  }

  void merge(int b, int c, int intensity, int x, int y) {
    // c < b
    if (c != b) {
      regionSumIntensity[c] += regionSumIntensity[b];
      regionSize[c] += (regionSize[b] + 1);

      for (int i = 0; i <= x; ++i)
        for (int j = 0; j <= y; ++j)
          if (arr[i][j] == b) arr[i][j] = c;

      count--;
    }
    this->addElem(intensity, x, y, c);
  }

  void print(cv::Mat& source, std::string alg) {
    cv::Mat ans = source.Mat::clone();
    for (int i = 1; i < n; ++i)
      for (int j = 1; j < m; ++j)
        if (arr[i][j] > 0) ans.at<uchar>(i, j) = 255;

    imwrite("Region.jpg", ans);
    imshow("Region", ans);
    cv::waitKey(0);
  }
};

void RegionGrowing(cv::Mat& source, double sigma, std::string alg) {
  regions r(source.rows, source.cols);
  double deltaB, deltaC, B, C;
  for (int x = 1; x < source.rows; ++x)
    for (int y = 1; y < source.cols; ++y) {
      int intensity = source.at<uchar>(x, y) ? 1 : 0;
      if (intensity > 0) {
        if (x == 1 && y == 1)
          B = C = 0;
        else {
          B = r.clavg(r.arr[x][y - 1]);
          C = r.clavg(r.arr[x - 1][y]);
        }
        intensity = 1;
        deltaB = abs(intensity - B);
        deltaC = abs(intensity - C);
        if (deltaB > sigma && deltaC > sigma) r.createRegion(intensity, x, y);
        if (deltaB <= sigma && deltaC > sigma)
          r.addElem(intensity, x, y, r.arr[x][y - 1]);
        if (deltaB > sigma && deltaC <= sigma)
          r.addElem(intensity, x, y, r.arr[x - 1][y]);
        if (deltaB <= sigma && deltaC <= sigma)
          if (abs(B - C) <= sigma)
            r.merge(r.arr[x][y - 1], r.arr[x - 1][y], intensity, x, y);
          else if (deltaB < deltaC)
            r.addElem(intensity, x, y, r.arr[x][y - 1]);
          else
            r.addElem(intensity, x, y, r.arr[x - 1][y]);
      }
    }
  std::cout << "\nCount regions" << alg << " = " << r.count;
  r.print(source, alg);
}

void OtsuTreshold(cv::Mat& source) {
  cv::Mat res = source.Mat::clone();
  std::vector<double> hist(256);
  int intensity;
  for (int i = 0; i < source.rows; i++)
    for (int j = 0; j < source.cols; j++) {
      intensity =
          round((source.at<cv::Vec3b>(i, j)[0] + source.at<cv::Vec3b>(i, j)[1] +
                 source.at<cv::Vec3b>(i, j)[2]) /
                3);
      hist[intensity]++;
    }
  int max = 0;         //максимальная интенсивность
  int min = INFINITY;  //минимальная интенсивность
  for (int i = 0; i < 256; i++) {
    max = (hist[i] > max) ? hist[i] : max;
    min = (hist[i] < min) ? hist[i] : min;
  }

  int maxi = 0, mini = 255;  //первая и последняя ненулевые интенсивности
  for (int i = 0; i < 256; i++)
    if (hist[i]) {
      mini = i;
      break;
    }

  for (int i = 255; i > -1; i--)
    if (hist[i]) {
      maxi = i;
      break;
    }

  cv::Mat histog = cv::Mat::zeros(102, 256, CV_8UC3);
  for (int i = 0; i < 256; i++) {
    int up = hist[i] / max * 100;
    for (int j = 100; j > 100 - up; j--)
      histog.at<cv::Vec3b>(j, i)[1] = histog.at<cv::Vec3b>(j, i)[0] = 200;
  }
  cv::imshow("Histogram", histog);
  cv::waitKey();

  int histSize = maxi - mini + 1;

  std::vector<int> histogram(histSize);
  for (int i = 0; i < histSize; i++) histogram[i] = hist[i + mini] - min;

  int m = 0;  //сумма высот столбиков*положение середины
  int n = 0;  //сумма высот столбиков

  for (int i = 0; i < histSize; i++) {
    m += i * histogram[i];
    n += histogram[i];
  }

  float maxSigma = -1;  //максимальное значение межклассовой дисперсии
  int treshold = 0;     //порог, соответствующий maxSigma
  int alpha1 = 0;       //сумма высот для класса 1
  int beta1 = 0;  //сумма высот*положение середины для класса 1

  for (int t = 0; t < histSize - 1; t++)  //по всем значениям порога
  {
    alpha1 += t * histogram[t];
    beta1 += histogram[t];

    float w1 = (float)beta1 / n;  //вероятность класса 1
    float a = (float)alpha1 / beta1 -
              (float)(m - alpha1) /
                  (n - beta1);  //разность средних арифметических классов 1 и 2
    float sigma = w1 * (1 - w1) * a * a;  //межклассовая дисперсия

    if (sigma > maxSigma) {
      maxSigma = sigma;
      treshold = t;
    }
  }
  treshold += min;

  for (int i = 0; i < source.rows; i++)
    for (int j = 0; j < source.cols; j++) {
      int intens = source.at<cv::Vec3b>(i, j)[0] +
                   source.at<cv::Vec3b>(i, j)[1] +
                   source.at<cv::Vec3b>(i, j)[2];
      res.at<cv::Vec3b>(i, j) = (intens >= treshold * 3)
                                    ? cv::Vec3b(255, 255, 255)
                                    : cv::Vec3b(0, 0, 0);
    }

  imwrite("Otsu.jpg", res);
  imshow("Otsu", res);
  cv::waitKey(0);

  RegionGrowing(res, 0.019, "Otsu");
}

void GaussFilter(cv::Mat& image, cv::Mat& result, int radius, double sigma) {
  double** kernel = new double*[radius];
  for (int i = 0; i < radius; i++) kernel[i] = new double[radius];

  double sum = 0;

  for (int x = -radius / 2; x <= radius / 2; ++x) {
    for (int y = -radius / 2; y <= radius / 2; ++y) {
      kernel[x + radius / 2][y + radius / 2] =
          (exp((x * x + y * y) / (2 * sigma * sigma * (-1.0)))) /
          (M_PI * 2 * sigma * sigma);
      sum += kernel[x + radius / 2][y + radius / 2];
    }
  }

  // нормализация
  for (int i = 0; i < radius; ++i)
    for (int j = 0; j < radius; ++j) kernel[i][j] = kernel[i][j] / sum;

  uchar color;
  int tmp1, tmp2;

  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j) {
      color = 0;
      for (int p = 0; p < radius; ++p)
        for (int q = 0; q < radius; ++q) {
          tmp1 = i + p;
          while (tmp1 >= image.rows) tmp1--;
          tmp2 = j + q;
          while (tmp2 >= image.cols) tmp2--;
          color += kernel[p][q] * image.at<uchar>(tmp1, tmp2);
        }
      result.at<uchar>(i, j) = color;
    }

  for (int i = 0; i < radius; i++) delete[] kernel[i];
  delete[] kernel;

  imwrite("GaussFilter.jpg", result);
}

void SobelOperation(cv::Mat& image, cv::Mat& result, cv::Mat& resultGard) {
  int MGx[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
  int MGy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  int GX = 0, GY = 0, G = 0, T = 0;
  for (int y = 1; y < image.rows - 1; ++y)
    for (int x = 1; x < image.cols - 1; ++x) {
      GX = 0;
      GY = 0;
      G = 0;
      T = 0;
      for (int p = -1; p < 2; ++p)
        for (int q = -1; q < 2; ++q) {
          GX += image.at<uchar>(y + p, x + q) * MGx[p + 1][q + 1];
          GY += image.at<uchar>(y + p, x + q) * MGy[p + 1][q + 1];
        }
      G = sqrtl(GX * GX + GY * GY);
      if (G != 0)
        T = (int)(atan2(GX, GY) / (M_PI / 4)) * M_PI / 4 - M_PI / 2;
      else
        T = ErrCode;
      result.at<uchar>(y, x) = G;
      resultGard.at<uchar>(y, x) = T;
    }
}

int Sign(double Val) {
  if (Val == 0.) return 0;
  if (Val > 0.) return 1;
  return -1;
}

bool IsCorrectIndex(int rows, int cols, int x, int y) {
  if (x < 0 || y < 0 || x > rows - 1 || y > cols - 1) return 0;
  return 1;
}

void NonMaximumSuppression(cv::Mat& sobel, cv::Mat& gard, cv::Mat& nonmax) {
  int dx, dy;
  for (int i = 0; i < (sobel.rows - 1); ++i)
    for (int j = 0; j < (sobel.cols - 1); ++j) {
      if (gard.at<uchar>(i, j) == ErrCode) continue;

      dx = Sign(cos(gard.at<uchar>(i, j)));
      dy = -1 * Sign(sin(gard.at<uchar>(i, j)));

      if (IsCorrectIndex(sobel.rows, sobel.cols, i + dx, j + dy))
        if (sobel.at<uchar>(i + dx, j + dy) <= sobel.at<uchar>(i, j))
          nonmax.at<uchar>(i + dx, j + dy) = 0;

      if (IsCorrectIndex(sobel.rows, sobel.cols, i - dx, j - dy))
        if (sobel.at<uchar>(i - dx, j - dy) <= sobel.at<uchar>(i, j))
          nonmax.at<uchar>(i - dx, j - dy) = 0;

      nonmax.at<uchar>(i, j) = sobel.at<uchar>(i, j);
    }
}

void DoubleThresholding(cv::Mat& nonmax, cv::Mat& result, double low,
                        double hight) {
  int down = 255 * low;
  int up = 255 * hight;

  for (int i = 0; i < nonmax.rows - 1; ++i)
    for (int j = 0; j < nonmax.cols - 1; ++j) {
      if (nonmax.at<uchar>(i, j) >= up)
        result.at<uchar>(i, j) = 255;
      else if (nonmax.at<uchar>(i, j) <= down)
        result.at<uchar>(i, j) = 0;
      else
        result.at<uchar>(i, j) = 127;
    }
}

void Blob_Analysis(cv::Mat& thresh, cv::Mat& result, int hight, int clear) {
  int dx, dy, x, y;
  int MoveDir[2][8] = {{-1, -1, -1, 0, 0, 1, 1, 1},
                       {-1, 0, 1, -1, 1, -1, 0, 1}};

  for (int i = 0; i < (thresh.rows - 1); ++i)
    for (int j = 0; j < (thresh.cols - 1); ++j) {
      if (thresh.at<uchar>(i, j) == hight) {
        result.at<uchar>(i, j) = hight;
        for (int k = 0; k < 8; k++) {
          dx = MoveDir[0][k];
          dy = MoveDir[1][k];
          x = i;
          y = j;
          while (1) {
            x += dx;
            y += dy;

            if (x < 0 || y < 0 || x > thresh.rows - 1 || y > thresh.cols - 1)
              break;
            if (thresh.at<uchar>(x, y) == clear ||
                thresh.at<uchar>(x, y) == hight)
              break;
            result.at<uchar>(x, y) = hight;
          }
        }
      }
      result.at<uchar>(i, j) = clear;
    }
  result = thresh - result;
}

void CannyBorderDetection(cv::Mat& source) {
  //Преобразование в оттенки серого
  cv::Mat res = source.Mat::clone();
  cv::cvtColor(source, res, cv::COLOR_BGR2GRAY);
  imshow("cvtColor", res);
  cv::waitKey(0);

  //Сглаживание
  cv::Mat filter = res.Mat::clone();
  GaussFilter(res, filter, 7, 1.);
  imshow("GaussFilter", filter);
  cv::waitKey(0);

  //Поиск градиентов
  cv::Mat gard = filter.Mat::clone();
  cv::Mat sobel = filter.Mat::clone();
  SobelOperation(filter, sobel, gard);
  imshow("SobelOperation", sobel);
  cv::waitKey(0);

  //Подавление не-максимумов
  cv::Mat nonmax = sobel.Mat::clone();
  NonMaximumSuppression(sobel, gard, nonmax);
  imshow("NonMaximumSuppression", nonmax);
  cv::waitKey(0);

  //Двойная пороговая фильтрация
  cv::Mat thresh = sobel.Mat::clone();
  DoubleThresholding(nonmax, thresh, 0.5, 0.7);
  imshow("DoubleThresholding", thresh);
  cv::waitKey(0);
  imwrite("Canny.jpg", thresh);
}

void Watershed(cv::Mat& source) {
  //Создаем бинарное изображение из исходного
  cv::Mat bw;
  cv::cvtColor(source, bw, cv::COLOR_BGR2GRAY);
  cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY);

  //Выполняем алгоритм distance transform
  cv::Mat dist;
  distanceTransform(bw, dist, cv::DIST_L2, 3);

  //Нормализуем изображение в диапозоне {0.0 1.0}
  normalize(dist, dist, 0, 1., cv::NORM_MINMAX);

  //Выполняем Threshold для определения пиков
  //Это будут маркеры для объектов на переднем плане
  threshold(dist, dist, .5, 1., cv::THRESH_BINARY);

  //Создаем CV_8U версию distance изображения
  //Это нужно для фуекции cv::findContours()
  cv::Mat dist_8u;
  dist.convertTo(dist_8u, CV_8U);

  // Находим все маркеры
  std::vector<std::vector<cv::Point> > contours;
  findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  auto ncomp = static_cast<int>(contours.size());

  // Создаем маркерное изображение для алгоритма watershed
  cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);

  // Рисуем маркеры переднего плана
  for (int i = 0; i < ncomp; i++)
    drawContours(markers, contours, i, cv::Scalar::all(i + 1), -1);

  // Рисуем маркеры фона
  circle(markers, cv::Point(5, 5), 3, CV_RGB(255, 255, 255), -1);

  // Выполняем алгоритм watershed
  watershed(source, markers);

  // Результирующее изображение
  cv::Mat res = cv::Mat::zeros(markers.size(), CV_8UC3);
  for (int i = 0; i < markers.rows; i++) {
    for (int j = 0; j < markers.cols; j++) {
      int index = markers.at<int>(i, j);
      if (index > 0 && index <= ncomp)
        res.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
      else
        res.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
    }
  }

  imwrite("Watershed.jpg", res);
  imshow("Watershed", res);
  cv::waitKey(0);

  RegionGrowing(res, 0.011, "Watershed");
}
