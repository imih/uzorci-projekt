#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TextBlock.h"

#define TRACE(x) std::cout << #x << " = " << x << std::endl

using namespace cv;
using namespace std;
using texture::TextBlock;


const int kTexBlockSize = 16;
const int kColTexBinSize = 16;
const int kColMax = 255;
const int dx[] = {0, -1, -1,  1}; //P0, P90, P45, P135
const int dy[] = {1,  0,  1, -1};

double round(double x) {
  if((int) (x + 0.5)  != (int) x)
    return (int) x + 1;
  else return (int) x;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


Mat getCooccur(Mat a, int p, int d) {
  Mat ret(kColTexBinSize + 1, kColTexBinSize + 1, CV_32S , 0);
  int n = ret.cols;
  int m = ret.rows;
  for(int i = 0; i < n; ++i) {
    int x = i + d * dx[p];
    if(x < 0 || x >= n) 
      continue;
    for(int j = 0; j < m; ++j) {
      int y = j + d * dy[p];
      if(y < 0 || y >= m)
        continue;
      int val1 = a.at<unsigned char>(i, j);
      int val2 = a.at<unsigned char>(x, y);
      ret.at<int>(val1, val2) += 1;
      ret.at<int>(val2, val1) += 1;
    }
  }
  return ret;
}

vector<double> getTextBlocks(char* srcImage, int p) {
  Mat image = cv::imread(srcImage, 1); //B, G, R
  cv::cvtColor(image, image, CV_BGR2HSV); //convertinje u hsv
  Mat m[3];
  cv::split(image, m); // split image channels to 3 matrices m[0], m[1], m[2]
  TRACE(type2str(image.type()));

  //16 bin color bounds
  for(int i = 0; i < image.cols; ++i)
    for(int j = 0; j < image.rows; ++j) {
      for(int k = 0; k < image.channels(); ++k) {
        unsigned char val = m[k].at<unsigned char>(i, j);
        m[k].at<unsigned char>(i, j) = (unsigned char) round(
            (double) val / kColMax * kColTexBinSize);
      }
    }

  vector<double> features;
  // co-occ matrix for channel k on distance d of orientation p
  for(int i = 0; i < image.cols; i += (kTexBlockSize / 2) + 1)  {
    for(int j = 0; j < image.rows; j += (kTexBlockSize / 2) + 1) {
      for(int k = 0; k < image.channels(); ++k) {
        Mat blockIm = Mat(m[k], Rect(i, j,
              min(kTexBlockSize, image.cols - i), 
              min(kTexBlockSize, image.rows - j)));
        for(int d = 1; d < kColTexBinSize; ++d) {
          Mat com = getCooccur(blockIm, p, d);
          TextBlock t(com);
          vector<double> curF = t.toFeatures();
          features.insert(features.end(), curF.begin(), curF.end());
        }
      }
    }
  }
  return features;
}

  int main(int argc, char** argv) {
    if(argc != 2) {
      puts("Fali slika!\n");
      exit(1);
    }

    puts("Texture features extraction\n");
    for(int k = 0; k <  4; ++k)
      getTextBlocks(argv[1], k);
    puts("Done\n");

    return 0;
  }
