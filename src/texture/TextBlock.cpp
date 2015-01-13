#include "TextBlock.h"

#include <cassert>

namespace texture {
  using cv::Mat;

  TextBlock::TextBlock(int p, int bl_id) {
    coOccType = p;
    block_id = bl_id;
    idx = 0;
  }

  void TextBlock::addFeatures(cv::Mat& blockIm) {
    const int dx[] = {0, -1, -1,  1}; //P0, P90, P45, P135
    const int dy[] = {1,  0,  1, -1};

    //Matrix of floats 
    Mat cooc = Mat::zeros(kColTexBinSize + 1, kColTexBinSize + 1, CV_32F);
    int n = cooc.rows;
    int m = cooc.cols;

    int r = blockIm.rows;
    int c = blockIm.cols;
    int d = 1;
    for(int i = 0; i < r; ++i) {
      int x = i + d * dx[coOccType];
      if(x < 0 || x >= r) 
        continue;
      for(int j = 0; j < c; ++j) {
        int y = j + d * dy[coOccType];
        if(y < 0 || y >= c)
          continue;
        int val1 = (int) blockIm.at<unsigned char>(i, j);
        int val2 = (int) blockIm.at<unsigned char>(x, y);
        assert(val1 >= 0 && val1 <= kColTexBinSize);
        assert(val2 >= 0 && val2 <= kColTexBinSize);
        cooc.at<float>(val1, val2) += 1;
        cooc.at<float>(val2, val1) += 1;
      }
    }

    //input: Matrix of floats
    int ng = cooc.rows - 1; // n = ng + 1
    long long  R = 0;
    for(int i = 1; i <= ng; ++i)
      for(int j = 1; j <= ng; ++j)
        R += cooc.at<float>(i, j);

    Mat p = cooc.clone();
    Mat px = Mat::zeros(1, ng + 1, CV_32F);
    Mat py = Mat::zeros(1, ng + 1, CV_32F);
    Mat pxpy = Mat::zeros(1, 2 * ng + 1, CV_32F);
    Mat pxmy = Mat::zeros(1, ng, CV_32F);

    vector<float> f_ = vector<float>(13);
    for(int i = 1; i <= ng; ++i) {
      for(int j = 1; j <= ng; ++j) {
        p.at<float>(i, j) /= R;
        px.at<float>(0, i) += p.at<float>(i, j);
        py.at<float>(0, j) += p.at<float>(i, j);
        pxpy.at<float>(0, i + j) +=  p.at<float>(i, j);
        pxmy.at<float>(0, abs(i -j)) += p.at<float>(i, j);
        f_[0] += (p.at<float>(i, j) * p.at<float>(i, j));
        f_[2] += (p.at<float>(i, j) * i * j);
        f_[4] += (p.at<float>(i, j) / (1 + (i - j) * (i - j)));
        if(fabs(p.at<float>(i, j) - 10e-6) > 10e-6)
          f_[8] -= (p.at<float>(i, j) * log(p.at<float>(i, j)));
      }
    }

    float mean_val = mean(p)[0];
    float hx = 0, hy = 0;
    for(int i = 1; i <= ng; ++i) {
      f_[3] += (px.at<float>(0, i) * (i - mean_val));
      if((fabs(px.at<float>(0, i)) - 10e-6) > 10e-6)
        hx += (px.at<float>(0, i) * log(px.at<float>(0, i)));
      if(fabs(py.at<float>(0, i) - 10e-6) > 10e-6)
        hy += (py.at<float>(0, i) * log(py.at<float>(0, i)));
    }

    cv::Scalar sigmax, sigmay, mix, miy;
    meanStdDev(px, mix, sigmax);
    meanStdDev(py, miy, sigmay);
    f_[2] = (f_[2] - mix[0] * miy[0]) / (sigmax[0] * sigmay[0]);

    float meanpxmy = mean(pxmy)[0];
    for(int k = 0; k <= ng - 1; ++k) {
      f_[1] += (pxmy.at<float>(0, k) * k * k);
      f_[9] += ((pxmy.at<float>(0, k) - meanpxmy) * (pxmy.at<float>(0, k) - meanpxmy));
      if(fabs(pxmy.at<float>(0, k) - 10e-6) > 10e-6)
        f_[10] -= (pxmy.at<float>(0, k) * log(pxmy.at<float>(0, k)));
    }
    f_[9]  /= ng;

    for(int i = 2; i <= 2 * ng; ++i) {
      f_[5] += (pxpy.at<float>(0, i) * i);
      if(fabs(pxpy.at<float>(0, i) - 10e-6) > 10e-6)
        f_[7] -= (pxpy.at<float>(0, i) * log(pxpy.at<float>(0, i)));
    }

    for(int i = 2; i <= 2 * ng; ++i) {
      f_[6] += ((float) (i - f_[7]) * (i - f_[7]) * pxpy.at<float>(0, i));
    }

    float hxy1 = 0, hxy2 = 0;
    for(int i = 1; i <= ng; ++i)
      for(int j = 1; j <= ng; ++j) {
        hxy1 -= (p.at<float>(i, j) * log(
              px.at<float>(0, i) * py.at<float>(0, j)));
        hxy2 -= (px.at<float>(0, i) * py.at<float>(0, j) * log(
              px.at<float>(0, i) * py.at<float>(0, j)));
      }

    f_[11] = (f_[8] - hxy1) / max(hx, hy);
    f_[12] = sqrt(1 - exp(-2 * hxy2 - f_[8]));

    if(f.n == 0)
      f = Vector<float>(3 * (int) f_.size());
    for(int i = 0; i < (int) f_.size(); ++i)
      f.SetElement(idx + i, f_[i]);
    idx += (int) f_.size();
  }
}

