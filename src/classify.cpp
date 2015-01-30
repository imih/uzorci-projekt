#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "texture/TextFeatures.h" 
#include "hogcolor/HOGAndColorFeatures.h"
#include "pls/plsanje.h"

#include "pls/maths.h"

using namespace texture;
using namespace hog;
using namespace std;
using namespace cv;

#define TRACE(x) std::cout << #x << " = " << x << std::endl

vector<TextBlock > getTextFeatures(Mat image) {
  vector<TextBlock> ret;
  getTextBlocks(image, ret);
  return ret;
}

vector<HOGBlock> getHOGFeatures(Mat image) {
  vector<HOGBlock> ret;
  calc_features(image, ret);
  return ret;
} 

int main(int argc, char** argv) { //have posImNodes and negImNodes now
  vector<double> scales = {0.4, 0.6, 0.8, 1.0, 1.2, 1.4};
  const int colWin = 64;
  const int rowWin = 128;

  puts("Getting features...pos...");
  clock_t begin = clock();

  Mat image = cv::imread(argv[argc - 1]);
  assert(image.rows > 0);
  assert(image.cols > 0);
  vector<Rect> rectangles;

  Model model("plsModel2");
  CvSVM svm;
  svm.load("svmModel2.xml");

  for(double s : scales) {
    Mat curScale;
    int newRows = (double) image.rows * s;
    int newCols = (double) image.cols * s;
    printf("%d %d\n", newRows, newCols);
    resize(image, curScale, Size(newCols, newRows), 0, 0, INTER_CUBIC);
    for(int i = 0; i + rowWin <= newRows; i += max(8, (int) s * 8)) {
      for(int j = 0; j + colWin <= newCols; j += max(8, (int) s * 8)) {
        Mat curWin(curScale, Rect(j, i, colWin, rowWin));
        vector<TextBlock> tex = getTextFeatures(curScale);
        vector<HOGBlock> hog = getHOGFeatures(curScale);
        vector<float> feats = getFeats(tex, hog, (int) tex.size() + (int) hog.size(), false);
        Matrix<float> data(1, (int) feats.size());
        for(int j = 0; j < (int) feats.size(); ++j)
          data.SetValue(0, j, feats[j]);
        Matrix<float>* proj = model.ProjectFeatureMatrix(&data);
        Mat* m = ConvertMatrixMat(proj);
        float h = svm.predict(*m);
        if((int) h == 1) {
          printf("%lf %d %d\n", s, j, i);
          int ix = (double) i / s;
          int jx = (double) j / s;
          rectangles.push_back(Rect(jx, ix, (double) colWin / s, (double) rowWin / s));
          //s, j, i 
        }

        curWin.release();
        delete proj;
        delete m;
      }
    }
  }

  cv::groupRectangles(rectangles, 2);
  for(Rect& r : rectangles) {
    cv::rectangle(image, r, cv::Scalar(100, 100, 200), 2, CV_AA, 0);
  }

  imshow("Classification", image);
  cvWaitKey(0);


  puts("done.");
  return 0;
}
