#include "Processing.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include "SSIM.h"

void calcNeighbourhood(int x, int y, int n, int cols, int rows, int& xleft,
                       int& yleft, int& xright, int& yright) {
  if (x - n < 0) {
    xleft = 0;
    xright = 2 * n;
  } else if (x + n >= cols) {
    xright = cols - 1;
    xleft = cols - 2 * n - 1;
  } else {
    xright = x + n;
    xleft = x - n;
  }
  if (y - n < 0) {
    yleft = 0;
    yright = 2 * n;
  } else if (y + n >= rows) {
    yright = rows - 1;
    yleft = rows - 2 * n - 1;
  } else {
    yright = y + n;
    yleft = y - n;
  }
}

double expon(cv::Mat& im, int x1, int y1, int x2, int y2, int k) {
  double h_squared, evk;
  evk = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
  h_squared = im.at<cv::Vec3b>(x1, y1)[k] - im.at<cv::Vec3b>(x2, y2)[k];
  if (h_squared == 0) h_squared = 0.1;
  return exp(-evk / (h_squared * h_squared));
}

void insertionSort(cv::Vec3b* window, int size) {
  int i, j;
  cv::Vec3b temp;
  for (i = 0; i < size * size; i++) {
    temp = window[i];
    for (j = i - 1; j >= 0 && (0.3 * temp.val[2] + 0.59 * temp.val[1] +
                               0.11 * temp.val[0]) < (0.3 * window[j].val[2] +
                                                      0.59 * window[j].val[1] +
                                                      0.11 * window[j].val[0]);
         j--) {
      window[j + 1] = window[j];
    }
    window[j + 1] = temp;
  }
}

void Dilation(cv::Mat& image, cv::Mat& res) {
  res = image;

  int** mat = new int*[image.rows];
  for (int i = 0; i < image.rows; i++) {
    mat[i] = new int[image.cols];
    for (int j = 0; j < image.cols; j++) {
      if (image.at<cv::Vec3b>(i, j)[0] == 255)
        mat[i][j] = 1;
      else
        mat[i][j] = 0;
    }
  }

  for (int i = 2; i < image.rows - 2; i++)
    for (int j = 2; j < image.cols - 2; j++) {
      if (mat[i][j] == 1) {
        res.at<cv::Vec3b>(i + 1, j) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i - 1, j) == cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i, j + 1) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i, j - 1) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i + 1, j + 1) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i - 1, j + 1) == cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i + 1, j - 1) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i - 1, j - 1) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i + 2, j) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i - 2, j) == cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i, j + 2) = cv::Vec3b(255, 255, 255);
        res.at<cv::Vec3b>(i, j - 2) = cv::Vec3b(255, 255, 255);
      }
    }
}

void Erosion(cv::Mat& image, cv::Mat& res) {
  res = image;

  int** mat = new int*[image.rows];
  for (int i = 0; i < image.rows; i++) {
    mat[i] = new int[image.cols];
    for (int j = 0; j < image.cols; j++) {
      if (image.at<cv::Vec3b>(i, j)[0] == 255)
        mat[i][j] = 1;
      else
        mat[i][j] = 0;
    }
  }

  for (int i = 2; i < image.rows - 2; i++)
    for (int j = 2; j < image.cols - 2; j++) {
      if (mat[i][j] == 0) {
        res.at<cv::Vec3b>(i + 1, j) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i - 1, j) == cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i, j + 1) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i, j - 1) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i + 1, j + 1) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i - 1, j + 1) == cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i + 1, j - 1) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i - 1, j - 1) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i + 2, j) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i - 2, j) == cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i, j + 2) = cv::Vec3b(0, 0, 0);
        res.at<cv::Vec3b>(i, j - 2) = cv::Vec3b(0, 0, 0);
      }
    }
}

void Binarization(cv::Mat& image, cv::Mat& result) {
  double p = 0;
  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++)
      for (int k = 0; k < 3; k++) p += image.at<cv::Vec3b>(i, j)[k];
  p /= 3 * image.cols * image.rows;

  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++)
      if (image.at<cv::Vec3b>(i, j)[0] + image.at<cv::Vec3b>(i, j)[1] +
              image.at<cv::Vec3b>(i, j)[2] <
          p * 3)
        result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
      else
        result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
}

int MinMod(int a, int b, int c, int d) {
  int mm = MIN(abs(a), abs(b), abs(c), abs(d));
  if (mm == -a || mm == -b || mm == -c || mm == -d) return -mm;
  return mm;
}

void MedianFilter(cv::Mat& image, cv::Mat& result, int n) {
  int rows = image.cols, cols = image.rows;
  double w, res = 0, norm = 0;
  for (int k = 0; k < 3; k++)  //для каждой компоненты цвета
    for (int i1 = 0; i1 < cols - 1; i1++) {
      for (int j1 = 0; j1 < rows - 1; j1++)  //обход по пикселям
      {
        for (int i2 = i1 - n; i2 < i1 + n + 1; i2++) {
          if (i2 < 0 || i2 >= cols) continue;
          for (int j2 = j1 - n; j2 < j1 + n + 1; j2++) {
            if ((i1 == i2 && j1 == j2) || j2 < 0 || j2 >= rows) continue;
            w = expon(image, i1, j1, i2, j2, k);
            norm += w;
            res += image.at<cv::Vec3b>(i2, j2).val[k] * w;
          }
        }
        result.at<cv::Vec3b>(i1, j1).val[k] = res / norm;
        res = 0;
        norm = 0;
      }
    }
  imwrite("Median.jpg", result);
}

void MeanFilter(cv::Mat& image, cv::Mat& result) {
  for (int i = 1; i < image.rows - 1; i++)
    for (int j = 1; j < image.cols - 1; j++)
      for (int k = 1; k < 3; k++)
        result.at<cv::Vec3b>(i, j)[k] =
            (image.at<cv::Vec3b>(i, j)[k] + image.at<cv::Vec3b>(i - 1, j)[k] +
             image.at<cv::Vec3b>(i + 1, j)[k] +
             image.at<cv::Vec3b>(i - 1, j - 1)[k] +
             image.at<cv::Vec3b>(i, j - 1)[k] +
             image.at<cv::Vec3b>(i + 1, j - 1)[k] +
             image.at<cv::Vec3b>(i - 1, j + 1)[k] +
             image.at<cv::Vec3b>(i, j + 1)[k] +
             image.at<cv::Vec3b>(i + 1, j + 1)[k]) /
            9;
  imwrite("Mean.jpg", result);
}

void GaussFilter(cv::Mat& image, cv::Mat& result, int radius, double sigma) {
  double** kernel = new double*[radius];
  for (int i = 0; i < radius; i++) kernel[i] = new double[radius];

  if (sigma <= 0.000000001) sigma = 1.398;

  double sum = 0;

  for (int x = -radius / 2; x <= radius / 2; x++) {
    for (int y = -radius / 2; y <= radius / 2; y++) {
      kernel[x + radius / 2][y + radius / 2] =
          (exp((x * x + y * y) / (2 * sigma * sigma * (-1.0)))) /
          (M_PI * 2 * sigma * sigma);
      sum += kernel[x + radius / 2][y + radius / 2];
    }
  }

  for (int i = 0; i < radius; i++)
    for (int j = 0; j < radius; j++)
      kernel[i][j] = kernel[i][j] / sum;  // нормализация

  cv::Vec3b color;

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      color = 0;
      for (int p = 0; p < radius; p++) {
        for (int q = 0; q < radius; q++) {
          int tmp1 = i + p;
          while (tmp1 >= image.rows) tmp1--;
          int tmp2 = j + q;
          while (tmp2 >= image.cols) tmp2--;
          color += kernel[p][q] * image.at<cv::Vec3b>(tmp1, tmp2);
        }
      }
      result.at<cv::Vec3b>(i, j) = color;
    }
  }

  for (int i = 0; i < radius; i++) delete[] kernel[i];
  delete[] kernel;

  imwrite("GaussFilter.jpg", result);
}

void MorphOpening(cv::Mat& image, cv::Mat& result) {
  cv::Mat temp;
  Erosion(image, temp);
  Dilation(temp, result);
  imwrite("Opening.jpg", result);
}

void MorphClosing(cv::Mat& image, cv::Mat& result) {
  cv::Mat temp;
  Dilation(image, temp);
  Erosion(temp, result);
  imwrite("Closing.jpg", result);
}

void MidpointFilter(cv::Mat& image, cv::Mat& result, int size) {
  int n = size * size;
  cv::Vec3b* window = new cv::Vec3b[n];

  int i, j;
  int height = image.rows;
  int width = image.cols;

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      int x = -1;
      for (int p = 0; p < size; p++) {
        for (int q = 0; q < size; q++) {
          int tmp1 = i + p;
          while (tmp1 >= height) tmp1--;
          int tmp2 = j + q;
          while (tmp2 >= width) tmp2--;
          window[++x] = image.at<cv::Vec3b>(tmp1, tmp2);
        }
      }
      insertionSort(window, size);
      cv::Vec3b min = window[0];
      cv::Vec3b max = window[size - 1];
      cv::Vec3b avg = (min + max) / 2;
      result.at<cv::Vec3b>(i, j) = avg;
    }
  }
  delete[] window;
  imwrite("Midpoint.jpg", result);
}
