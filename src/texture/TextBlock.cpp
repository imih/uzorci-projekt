#include "TextBlock.h"

#include <cassert>

namespace texture {
  using cv::Mat;

  TextBlock::TextBlock(int p, int block_id) {
    coOccType = p;
    blockId = block_id;
  }

  void TextBlock::createFeatures() {
    if(f.n) return;
    f = Vector<float>((int) texF.size());
    for(int i = 0; i < texF.size(); ++i) {
      f.SetElement(i, texF[i]);
    }
    texF.clear();
  }

  Vector<float> TextBlock::getFeatures() {
    if(!f.n)
      createFeatures();
    return f;
  }

  vector<float> TextBlock::calcHaralick(Mat& blockIm, int d = 1) {
    const int dx[] = {0, -1, -1,  1}; //P0, P90, P45, P135
    const int dy[] = {1,  0,  1, -1};

    //Matrix of floats 
    Mat cooc = Mat::zeros(kColTexBinSize + 1, kColTexBinSize + 1, CV_32F);
    int n = cooc.rows;
    int m = cooc.cols;

    int r = blockIm.rows;
    int c = blockIm.cols;
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

    vector<float> f = vector<float>(13);
    for(int i = 1; i <= ng; ++i) {
      for(int j = 1; j <= ng; ++j) {
        p.at<float>(i, j) /= R;
        px.at<float>(0, i) += p.at<float>(i, j);
        py.at<float>(0, j) += p.at<float>(i, j);
        pxpy.at<float>(0, i + j) +=  p.at<float>(i, j);
        pxmy.at<float>(0, abs(i -j)) += p.at<float>(i, j);
        f[0] += (p.at<float>(i, j) * p.at<float>(i, j));
        f[2] += (p.at<float>(i, j) * i * j);
        f[4] += (p.at<float>(i, j) / (1 + (i - j) * (i - j)));
        f[8] -= (p.at<float>(i, j) * log(p.at<float>(i, j)));
      }
    }

    float mean_val = mean(p)[0];
    float hx = 0, hy = 0;
    for(int i = 1; i <= ng; ++i) {
      f[3] += (px.at<float>(0, i) * (i - mean_val));
      hx += (px.at<float>(0, i) * log(px.at<float>(0, i)));
      hy += (py.at<float>(0, i) * log(py.at<float>(0, i)));
    }

    cv::Scalar sigmax, sigmay, mix, miy;
    meanStdDev(px, mix, sigmax);
    meanStdDev(py, miy, sigmay);
    f[2] = (f[2] - mix[0] * miy[0]) / (sigmax[0] * sigmay[0]);

    float meanpxmy = mean(pxmy)[0];
    for(int k = 0; k <= ng - 1; ++k) {
      f[1] += (pxmy.at<float>(0, k) * k * k);
      f[9] += ((pxmy.at<float>(0, k) - meanpxmy) * (pxmy.at<float>(0, k) - meanpxmy));
      f[10] -= (pxmy.at<float>(0, k) * log(pxmy.at<float>(0, k)));
    }
    f[9]  /= ng;

    for(int i = 2; i <= 2 * ng; ++i) {
      f[5] += (pxpy.at<float>(0, i) * i);
      f[7] -= (pxpy.at<float>(0, i) * log(pxpy.at<float>(0, i)));
    }

    for(int i = 2; i <= 2 * ng; ++i) {
      f[6] += ((i - f[7]) * (i - f[7]) * pxpy.at<float>(0, i));
    }

    float hxy1 = 0, hxy2 = 0;
    for(int i = 1; i <= ng; ++i)
      for(int j = 1; j <= ng; ++j) {
        hxy1 -= (p.at<float>(i, j) * log(
              px.at<float>(0, i) * py.at<float>(0, j)));
        hxy2 -= (px.at<float>(0, i) * py.at<float>(0, j) * log(
              px.at<float>(0, i) * py.at<float>(0, j)));
      }

    f[11] = (f[8] - hxy1) / max(hx, hy);
    f[12] = sqrt(1 - exp(-2 * hxy2 - f[8]));
    return f;
  }

  void TextBlock::addChannel(int k, cv::Mat& blockIm) {
    //for channel k make co-occur for every dist d
    vector<float> curH = calcHaralick(blockIm);
    texF.insert(texF.end(), curH.begin(), curH.end());
  }

}

