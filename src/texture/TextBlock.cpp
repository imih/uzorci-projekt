#include "TextBlock.h"

namespace texture {
  using namespace cv;

  TextBlock::TextBlock(Mat occ) {
    f.clear();
    long long  R = 0;
    int n = occ.rows;
    int m = occ.cols;
    for(int i = 0; i < n; ++i)
      for(int j = 0; j < m; ++j)
        R += occ.at<int>(i, j);

    //f1 - angular second moment
    double f1 = 0;
    //f3 - correlation
    double f3 = 0;
    Mat p(n, m, CV_64F, 0); //double
    for(int i = 0; i < n; ++i)
      for(int j = 0; j < m; ++j) {
        p.at<double>(i, j) = (double) occ.at<int>(i, j) / R;
        f1 += (p.at<double>(i, j) * p.at<double>(i, j));
        f3 += (i * j * p.at<double>(i, j));
      }

    double mix = 0; //TODO
    double miy = 0; //TODO
    double sigmax = 0; //TODO
    double sigmay = 0; //TODO

    f3 = (f3 -  (mix * miy)) / sigmax / sigmay;
    Mat pxmy; //TODO

    //f2 - contrast
    double f2 = 0; 
    for(int k = 0; k <  n - 1; ++k) {
      f2 += (pxmy.at<double>(k) * k * k );
    }

    double mi = 0; //TODO

    //f4 - variance
    //f5 - inverse difference moment
    double f4 = 0;
    double f5 = 0;
    for(int i = 1; i < n; ++i)
      for(int j = 1; j < n; ++j) {
        f4 += p.at<int>(i, j) * (i - mi) * (i - mi);
        f5 += p.at<int>(i, j) / (1 + (i - j) * (i - j));
      }
    //TODO
  }

  vector<double> TextBlock::toFeatures() {
    return f;
  }

}
