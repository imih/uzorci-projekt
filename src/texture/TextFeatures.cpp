#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "TextBlock.h"
#include "TextFeatures.h"

#define TRACE(x) std::cout << #x << " = " << x << std::endl

namespace {
  std::string type2str(int type) {
    std::string r;

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
};

namespace texture {
  using namespace cv;
  using namespace std;

  void getTextBlocks(Mat orgImage, vector<TextBlock>& texBlocks, set<int> skippedBlocks) {
    int block_id = -1;
    for(int bl_size = 16; bl_size <= 32; bl_size += 16) {
      for(int p = 0; p < 4; ++p) {
        Mat image;
        orgImage.copyTo(image);
        cv::cvtColor(image, image, CV_BGR2HSV); //convertinje u hsv
        Mat m[3];
        cv::split(image, m); // split image channels to 3 matrices m[0], m[1], m[2]

        //16 bin color bounds values 1....16
        for(int i = 0; i < m[0].rows; ++i)
          for(int j = 0; j < m[0].cols; ++j) {
            for(int k = 0; k < image.channels(); ++k) {
              double val = (double) m[k].at<unsigned char>(i, j);
              m[k].at<unsigned char>(i, j) = (unsigned char) min(
                  (int) (val / kColMax * kColTexBinSize) + 1, kColTexBinSize);
            }
          }

        // co-occ matrix for channel k on distance d of orientation p
        for(int i = 0; i < m[0].cols; i += (bl_size / 2) + 1)  { 
          for(int j = 0; j < m[0].rows; j += (bl_size / 2) + 1) {
            block_id++;
            if(skippedBlocks.count(block_id))
              continue;
            TextBlock t(p, block_id);
            for(int k = 0; k < image.channels(); ++k) {
              int len1 = min(bl_size, (int) m[k].cols - i);
              int len2 = min(bl_size, (int) m[k].rows - j);
              Mat blockIm(m[k], Rect(i, j, len1, len2));
              t.addFeatures(blockIm);
            }
            texBlocks.push_back(t);
          }
        }
      }
    }
  }

};
