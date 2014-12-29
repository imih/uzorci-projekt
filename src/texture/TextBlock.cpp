#include "TextBlock.h"

namespace texture {
  using namespace cv;
  using namespace std;

  TextBlock::TextBlock(int p) {
    coOccType = p;
  }

  void TextBlock::createFeatures() {
    if(!f.empty()) return;
    for(int i = 0; i < texF.size(); ++i) {
      //item.k item.d item.p
      vector<double> dubH = texF[i].getFeatures();
      f.insert(f.end(), dubH.begin(), dubH.end());
    }
  }

  vector<double> TextBlock::getFeatures() {
    if(f.empty())
      createFeatures();
    return f;
  }

  Haralick TextBlock::calcHaralick(cv::Mat& blockIm, int d) const {
    const int dx[] = {0, -1, -1,  1}; //P0, P90, P45, P135
    const int dy[] = {1,  0,  1, -1};
    int p = coOccType;

    //Matrix of doubles 
    Mat cooc = Mat::zeros(kColTexBinSize + 1, kColTexBinSize + 1, CV_64F);
    int n = cooc.rows;
    int m = cooc.cols;
    for(int i = 0; i < n; ++i) {
      int x = i + d * dx[p];
      if(x < 0 || x >= n) 
        continue;
      for(int j = 0; j < m; ++j) {
        int y = j + d * dy[p];
        if(y < 0 || y >= m)
          continue;
        int val1 = (int) blockIm.at<unsigned char>(i, j);
        int val2 = (int) blockIm.at<unsigned char>(x, y);
        cooc.at<double>(val1, val2) += 1;
        cooc.at<double>(val2, val1) += 1;
      }
    }

    return Haralick(cooc);
  }

  void TextBlock::addChannel(int k, cv::Mat& blockIm) {
    //for channel k make co-occur for every dist d
    for(int d = 1; d < kColTexBinSize; ++d) {
      texF.push_back(calcHaralick(blockIm, d));
    }
  }

  Haralick::Haralick(Mat& cooc) {
    //input: Matrix of doubles
    int ng = cooc.rows - 1; // n = ng + 1
    long long  R = 0;
    for(int i = 1; i <= ng; ++i)
      for(int j = 1; j <= ng; ++j)
        R += cooc.at<double>(i, j);

    Mat p = cooc.clone();
    Mat px = Mat::zeros(1, ng + 1, CV_64F);
    Mat py = Mat::zeros(1, ng + 1, CV_64F);
    Mat pxpy = Mat::zeros(1, 2 * ng + 1, CV_64F);
    Mat pxmy = Mat::zeros(1, ng, CV_64F);

    f = vector<double>(13);
    for(int i = 1; i <= ng; ++i) {
      for(int j = 1; j <= ng; ++j) {
        p.at<double>(i, j) /= R;
        px.at<double>(0, i) += p.at<double>(i, j);
        py.at<double>(0, j) += p.at<double>(i, j);
        pxpy.at<double>(0, i + j) +=  p.at<double>(i, j);
        pxmy.at<double>(0, abs(i -j)) += p.at<double>(i, j);
        f[0] += (p.at<double>(i, j) * p.at<double>(i, j));
        f[2] += (p.at<double>(i, j) * i * j);
        f[4] += (p.at<double>(i, j) / (1 + (i - j) * (i - j)));
        f[8] -= (p.at<double>(i, j) * log(p.at<double>(i, j)));
      }
    }

    double mean_val = mean(p)[0];
    double hx = 0, hy = 0;
    for(int i = 1; i <= ng; ++i) {
      f[3] += (px.at<double>(0, i) * (i - mean_val));
      hx += (px.at<double>(0, i) * log(px.at<double>(0, i)));
      hy += (py.at<double>(0, i) * log(py.at<double>(0, i)));
    }

    Scalar sigmax, sigmay, mix, miy;
    meanStdDev(px, mix, sigmax);
    meanStdDev(py, miy, sigmay);
    f[2] = (f[2] - mix[0] * miy[0]) / (sigmax[0] * sigmay[0]);

    double meanpxmy = mean(pxmy)[0];
    for(int k = 0; k <= ng - 1; ++k) {
      f[1] += (pxmy.at<double>(0, k) * k * k);
      f[9] += ((pxmy.at<double>(0, k) - meanpxmy) * (pxmy.at<double>(0, k) - meanpxmy));
      f[10] -= (pxmy.at<double>(0, k) * log(pxmy.at<double>(0, k)));
    }
    f[9]  /= ng;

    for(int i = 2; i <= 2 * ng; ++i) {
      f[5] += (pxpy.at<double>(0, i) * i);
      f[7] -= (pxpy.at<double>(0, i) * log(pxpy.at<double>(0, i)));
    }

    for(int i = 2; i <= 2 * ng; ++i) {
      f[6] += ((i - f[7]) * (i - f[7]) * pxpy.at<double>(0, i));
    }

    double hxy1 = 0, hxy2 = 0;
    for(int i = 1; i <= ng; ++i)
      for(int j = 1; j <= ng; ++j) {
        hxy1 -= (p.at<double>(i, j) * log(
              px.at<double>(0, i) * py.at<double>(0, j)));
        hxy2 -= (px.at<double>(0, i) * py.at<double>(0, j) * log(
              px.at<double>(0, i) * py.at<double>(0, j)));
      }

    f[11] = (f[8] - hxy1) / max(hx, hy);
    f[12] = sqrt(1 - exp(-2 * hxy2 - f[8]));
  }

  vector<double> Haralick::getFeatures() {
    return f;
  }

}

