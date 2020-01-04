#include "Fragmentation.h"
#include <cmath>

#define PI 3.14159265359

void FourierTransform(Mat& source, Mat& fourier, pair<float, float>** Fuv) {
  int n = source.rows;

  pair<float, float>** Fxv = new pair<float, float>*[n];
  pair<float, float>** FuvTemp = new pair<float, float>*[n];

  for (int i = 0; i < n; ++i) {
    Fxv[i] = new pair<float, float>[n];
    FuvTemp[i] = new pair<float, float>[n];
  }

  for (int x = 0; x < n; ++x)
    for (int v = 0; v < n; ++v)
      for (int y = 0; y < n; ++y) {
        Fxv[x][v].first += source.at<uchar>(x, y) * cos(2 * PI * v * y / n);
        Fxv[x][v].second -= source.at<uchar>(x, y) * sin(2 * PI * v * y / n);
      }

  float arg;
  for (int v = 0; v < n; ++v)
    for (int u = 0; u < n; ++u)
      for (int x = 0; x < n; ++x) {
        arg = 2 * PI * u * x / n;
        FuvTemp[u][v].first +=
            cos(arg) * Fxv[x][v].first + sin(arg) * Fxv[x][v].second;
        FuvTemp[u][v].second +=
            cos(arg) * Fxv[x][v].second - sin(arg) * Fxv[x][v].first;
      }

  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v) {
      FuvTemp[u][v].first /= n;
      FuvTemp[u][v].second /= n;
    }

  //Транспонируем по обоим диагоналям
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      Fuv[i][j].first = FuvTemp[(i + n / 2) % n][(j + n / 2) % n].first;
      Fuv[i][j].second = FuvTemp[(i + n / 2) % n][(j + n / 2) % n].second;
      fourier.at<uchar>(i, j) =
          sqrt(powf(Fuv[i][j].first, 2) + powf(Fuv[i][j].second, 2));
    }

  for (int i = 0; i < n; ++i) {
    delete[] Fxv[i];
    delete[] FuvTemp[i];
  }
  delete[] Fxv;
  delete[] FuvTemp;
}

void InverseFourierTransform(Mat& result, pair<float, float>** Fuv) {
  int n = result.rows;
  pair<float, float>** Iuy = new pair<float, float>*[n];
  pair<float, float>** Ixy = new pair<float, float>*[n];

  for (int i = 0; i < n; ++i) {
    Iuy[i] = new pair<float, float>[n];
    Ixy[i] = new pair<float, float>[n];
  }

  for (int u = 0; u < n; ++u)
    for (int y = 0; y < n; ++y)
      for (int v = 0; v < n; ++v) {
        Iuy[u][y].first += Fuv[u][v].first * cos(2 * PI * v * y / n) -
                           Fuv[u][v].second * sin(2 * PI * v * y / n);
        Iuy[u][y].second += Fuv[u][v].second * cos(2 * PI * v * y / n) +
                            Fuv[u][v].first * sin(2 * PI * v * y / n);
      }

  float arg;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x)
      for (int u = 0; u < n; ++u) {
        arg = 2 * PI * u * x / n;
        Ixy[x][y].first +=
            cos(arg) * Iuy[u][y].first - sin(arg) * Iuy[u][y].second;
        Ixy[x][y].second +=
            cos(arg) * Iuy[u][y].second + sin(arg) * Iuy[u][y].first;
      }

  for (int x = 0; x < n; ++x)
    for (int y = 0; y < n; ++y) {
      Ixy[x][y].first /= n;
      Ixy[x][y].second /= n;
      if (sqrt(powf(Ixy[x][y].first, 2) + powf(Ixy[x][y].second, 2)) > 255)
        result.at<uchar>(x, y) = 255;
      else
        result.at<uchar>(x, y) =
            sqrt(powf(Ixy[x][y].first, 2) + powf(Ixy[x][y].second, 2));
    }

  for (int i = 0; i < n; ++i) {
    delete[] Iuy[i];
    delete[] Ixy[i];
  }
  delete[] Iuy;
  delete[] Ixy;
}

// Фильтр низких частот - границы занулить(убираем шум)
// rad - радиус зануления границ
void LowPassFilter(pair<float, float>** Fuv, pair<float, float>** FuvRes, int n,
                   int rad, Mat& filter) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      FuvRes[i][j].first = FuvRes[i][j].second = 0;
      filter.at<uchar>(i, j) =
          sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
    }

  for (int i = rad; i < n - rad; ++i)
    for (int j = rad; j < n - rad; ++j) {
      FuvRes[i][j] = Fuv[i][j];
      filter.at<uchar>(i, j) =
          sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
    }
}

// Фильтр высоких частот - центр занулить(выделяем границы)
void HightPassFilter(pair<float, float>** Fuv, pair<float, float>** FuvRes,
                     int n, int size_obl, Mat& filter) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      FuvRes[i][j] = Fuv[i][j];
      filter.at<uchar>(i, j) =
          sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
    }

  int start = (n - size_obl) / 2;
  for (int i = start; i < start + size_obl; ++i)
    for (int j = start; j < start + size_obl; ++j) {
      FuvRes[i][j].first = FuvRes[i][j].second = 0;
      filter.at<uchar>(i, j) = 0;
    }
}

// Cчитаем ядро Фильтра Гаусса
void GaussFilter(double** kernel, int radFilter, double sigma) {
  double sum = 0;
  for (int x = -radFilter / 2; x <= radFilter / 2; x++)
    for (int y = -radFilter / 2; y <= radFilter / 2; y++) {
      kernel[x + radFilter / 2][y + radFilter / 2] =
          (exp((x * x + y * y) / (2 * sigma * sigma * (-1.0)))) /
          (PI * 2 * sigma * sigma);
      sum += kernel[x + radFilter / 2][y + radFilter / 2];
    }

  // Нормализация
  for (int i = 0; i < radFilter; i++)
    for (int j = 0; j < radFilter; j++) kernel[i][j] /= sum;
}

// Преобразование Фурье для Фильтра Гаусса
void FourierTransformFilter(double** kernel, pair<float, float>** Fkernel,
                            int radFilter) {
  pair<float, float>** Fxv = new pair<float, float>*[radFilter];
  pair<float, float>** FuvTemp = new pair<float, float>*[radFilter];

  for (int i = 0; i < radFilter; ++i) {
    Fxv[i] = new pair<float, float>[radFilter];
    FuvTemp[i] = new pair<float, float>[radFilter];
  }

  for (int x = 0; x < radFilter; ++x)
    for (int v = 0; v < radFilter; ++v)
      for (int y = 0; y < radFilter; ++y) {
        Fxv[x][v].first += kernel[x][y] * cos(2 * PI * v * y / radFilter);
        Fxv[x][v].second -= kernel[x][y] * sin(2 * PI * v * y / radFilter);
      }

  float arg;
  for (int v = 0; v < radFilter; ++v)
    for (int u = 0; u < radFilter; ++u)
      for (int x = 0; x < radFilter; ++x) {
        arg = 2 * PI * u * x / radFilter;
        FuvTemp[u][v].first +=
            cos(arg) * Fxv[x][v].first + sin(arg) * Fxv[x][v].second;
        FuvTemp[u][v].second +=
            cos(arg) * Fxv[x][v].second - sin(arg) * Fxv[x][v].first;
      }

  for (int u = 0; u < radFilter; ++u)
    for (int v = 0; v < radFilter; ++v) {
      FuvTemp[u][v].first /= radFilter;
      FuvTemp[u][v].second /= radFilter;
    }

  //Транспонируем по обеим диагоналям
  for (int i = 0; i < radFilter; ++i)
    for (int j = 0; j < radFilter; ++j) {
      Fkernel[i][j].first = FuvTemp[(i + radFilter / 2) % radFilter]
                                   [(j + radFilter / 2) % radFilter]
                                       .first;
      Fkernel[i][j].second = FuvTemp[(i + radFilter / 2) % radFilter]
                                    [(j + radFilter / 2) % radFilter]
                                        .second;
    }

  for (int i = 0; i < radFilter; ++i) {
    delete[] Fxv[i];
    delete[] FuvTemp[i];
  }
  delete[] Fxv;
  delete[] FuvTemp;
}

/*void SobelOperation(int** MGx, int** MGy) {
  int MGX[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
  int MGY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

 
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
        T = (int)(atan2(GX, GY) / (PI / 4)) * PI / 4 - PI / 2;
      result.at<uchar>(y, x) = G;
      resultGard.at<uchar>(y, x) = T;
    }
}*/

// Удаление шума фильтром Гаусса
void Gauss_Fourier_Filter(pair<float, float>** Fuv, pair<float, float>** FuvRes,
                          int n, int size_kernel, double sigma, Mat& filter) {
  /* Вычисление ядра фильтра Гаусса
   double** kernel = new double*[size_kernel];
   for (int i = 0; i < size_kernel; i++) kernel[i] = new double[size_kernel];
   GaussFilter(kernel, size_kernel, sigma);*/

  // Заполняем будущий фильтр нулями до изображения
  double** BigKernel = new double*[n];
  pair<float, float>** FBigKernel = new pair<float, float>*[n];
  for (int i = 0; i < n; ++i) {
    BigKernel[i] = new double[n];
    FBigKernel[i] = new pair<float, float>[n];
    for (int j = 0; j < n; ++j) BigKernel[i][j] = 0;
  }

  // Вставляем ядро в центр фильтра
  int start = (n - 3) / 2;
  for (int i = start; i < start + 3; ++i)
    for (int j = start; j < start + 3; ++j) {
      BigKernel[i][j] = 0;
    }

  BigKernel[start + 1][start + 1] = 1;


  // Преобразуем фильтр
  FourierTransformFilter(BigKernel, FBigKernel, n);

  // Покомпонентно перемножаем
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      FuvRes[i][j].first = (Fuv[i][j].first * FBigKernel[i][j].first +
                            Fuv[i][j].second * FBigKernel[i][j].second) * 255;
      FuvRes[i][j].second = (Fuv[i][j].second * FBigKernel[i][j].second +
                             Fuv[i][j].second * FBigKernel[i][j].first) * 255;
      filter.at<uchar>(i, j) =
          sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
    }

 // for (int i = 0; i < size_kernel; i++) delete[] kernel[i];
  for (int i = 0; i < n; i++) {
    delete[] BigKernel[i];
    delete[] FBigKernel[i];
  }
 // delete[] kernel;
  delete[] BigKernel;
  delete[] FBigKernel;
}

void Filter(pair<float, float>** FuvRes, int n, int gr, Mat& filter) {
  double a;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      a = sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
      if (a > gr) {
        FuvRes[i][j].first = FuvRes[i][j].second = 0;
        filter.at<uchar>(i, j) = 0;
      }
    }
}

// Удаление периодического шума
void PeriodicNoiseRemoval(pair<float, float>** Fuv, pair<float, float>** FuvRes,
                          int n, int gr, int size_kernel, Mat& filter) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      FuvRes[i][j].first = Fuv[i][j].first;
      FuvRes[i][j].second = Fuv[i][j].second;
      filter.at<uchar>(i, j) =
          sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
    }

  Filter(FuvRes, n, gr, filter);

  int start = (n - size_kernel) / 2;
  for (int i = start; i < start + size_kernel; ++i)
    for (int j = start; j < start + size_kernel; ++j) {
      FuvRes[i][j].first = Fuv[i][j].first;
      FuvRes[i][j].second = Fuv[i][j].second;
      filter.at<uchar>(i, j) =
          sqrt(powf(FuvRes[i][j].first, 2) + powf(FuvRes[i][j].second, 2));
    }
}
