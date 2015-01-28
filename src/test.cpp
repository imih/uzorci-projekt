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

using namespace texture;
using namespace hog;
using namespace std;
using namespace cv;

#define TRACE(x) std::cout << #x << " = " << x << std::endl

const int debugFeaturesNo = 20000;
const int maxlen = 2048;

// window size is fixed: 64 x 128 
const int colWin = 64;
const int rowWin = 128;
const int negSampleSize = min(debugFeaturesNo, 2000);

void blocksToFeatures(Mat& data, Mat& res,
    vector<vector<TextBlock> >&posTex, vector<vector<TextBlock> >&negTex,
    vector<vector<HOGBlock> >& posHog, vector<vector<HOGBlock> >& negHog) {
  int features = (posTex[0].size() * posTex[0][0].f.n + posHog[0].size() * posHog[0][0].f.n);
  assert(posTex[0][0].f.n > 0);
  assert(posHog[0][0].f.n > 0);
  int N = posTex.size() + negTex.size();

  data = Mat(N, features, CV_32F);
  res = Mat(N, 1, CV_32F);

  int blocks = (int) posHog[0].size() + (int) posTex[0].size();

  //jednak broj pozitivnih primjera
  for(int i = 0; i < posTex.size(); ++i) {
    res.at<float>(i, 0) = 1;
    vector<float> allFeatures = getFeats(posTex[i], posHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      data.at<float>(i, j) = allFeatures[j];
    }
  }

  int shift = (int) posTex.size();
  for(int i = 0; i < negTex.size(); ++i) {
    res.at<float>(shift+ i, 0) = -1;
    vector<float> allFeatures = getFeats(negTex[i], negHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      data.at<float>(shift + i, j) = allFeatures[j];
    }
  }
}

struct ImNode { 
  string fileName; 
  ImNode(string name) {
    fileName = name;
  }
};

vector<ImNode> posImNodes, negImNodes;

char temp[maxlen];
void getTestSet() {
  ifstream posList("../dataset/test_64x128_H96/pos.lst");
  while(posList.getline(temp, maxlen)) {
    posImNodes.push_back(ImNode("../dataset/" + string(temp)));
  }
  posList.close();
  printf("Read %d pos window samples\n", (int) posImNodes.size());

  ifstream negList("../dataset/test_64x128_H96/neg.lst");
  while(negList.getline(temp, maxlen)) {
    negImNodes.push_back(ImNode("../dataset/" + string(temp)));
  }
  negList.close();
  printf("Read %d neg window samples\n", (int) negImNodes.size()); 
}

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
  puts("Getting features...pos...");
  vector<vector<TextBlock> > perBlockPosTex;
  vector<vector<HOGBlock> > perBlockPosHog;
  vector<vector<TextBlock> > perBlockNegTex; 
  vector<vector<HOGBlock> > perBlockNegHog;
  clock_t begin = clock();

  getTestSet();
  Mat temp = Mat(cv::imread(posImNodes[0].fileName, 1), Rect(0, 0, colWin, rowWin));
  for(int im = 0; im < min(debugFeaturesNo, (int) posImNodes.size()); ++im) {
    Mat curWin = cv::imread(posImNodes[im].fileName, 1);
    cv::resize(curWin, curWin, temp.size(), 0, 0, INTER_CUBIC);
    assert(curWin.rows ==  rowWin);
    assert(curWin.cols ==  colWin);
    perBlockPosTex.push_back(getTextFeatures(curWin));
    perBlockPosHog.push_back(getHOGFeatures(curWin)); 
    if(im % 10 == 0) {
      clock_t endPos = clock();
      printf("%d/%d t:%0.3lfs\n", im + 1, (int) posImNodes.size(),
          double(endPos - begin) / CLOCKS_PER_SEC);
    }
    curWin.release();
  }

  puts("loading neg samples...");
  int w = 0;
  vector<cv::Mat> rest;
  for(int im = 0; (im < (int) negImNodes.size()); ++im) {
    Mat image = cv::imread(negImNodes[im].fileName, 1); //BGR
    if(image.rows < rowWin || image.cols < colWin)
      cv::resize(image, image, temp.size(), 0, 0, INTER_CUBIC);
    for(int i = 0; (i + rowWin <= image.rows); i += rowWin) 
      for(int j = 0; (j + colWin <= image.cols); j += colWin) {
        w++;
        Mat curWin = Mat(image, Rect(j, i, colWin, rowWin));
        assert(curWin.rows ==  rowWin);
        assert(curWin.cols ==  colWin);
        perBlockNegTex.push_back(getTextFeatures(curWin));
        perBlockNegHog.push_back(getHOGFeatures(curWin)); 
        if(im % 10 == 0) {
          clock_t endPos = clock();
          printf("%d/%d t:%0.3lfs\n", im + 1, negSampleSize, 
              double(endPos - begin) / CLOCKS_PER_SEC);
        }
        curWin.release();
        if(w > negSampleSize) break;
      } 
    image.release();
    if(w > negSampleSize) break;
  }

  clock_t endPos = clock();
  printf("%d/oo t:%0.3lfs\n", w, double(endPos - begin) / CLOCKS_PER_SEC);
  puts("...done loading  features");

  puts("Loading models...");
  Model model("plsModel2");
  CvSVM svm;
  svm.load("svmModel2.xml");

  puts("Performing evaluation...");
  Mat data, res;
  blocksToFeatures(data, res, perBlockPosTex, perBlockNegTex, 
      perBlockPosHog, perBlockNegHog);
  perBlockPosTex.clear();
  perBlockPosHog.clear();
  perBlockNegTex.clear();
  perBlockNegHog.clear();

  Matrix<float> *mData = ConvertMatMatrix(data);
  Matrix<float> *mProj = model.ProjectFeatureMatrix(mData);
  Mat *newData = ConvertMatrixMat(mProj);
  cv::Mat valH = Mat(newData->rows, 1, CV_32F);
  for(int j = 0; j < newData->rows; ++j)
    valH.at<float>(j, 0) = svm.predict(newData->row(j), false);
  delete mData;
  delete mProj;
  delete newData;

  int n = valH.rows;
  int fp = 0, tn = 0, fn = 0, tp = 0;
  FILE *f = fopen("fails", "w");
  for(int i = 0; i  < n; ++i) {
    int yk = (int) res.at<float>(i, 0);
    int hk = (int) valH.at<float>(i, 0);
    fp += (hk == 1 && yk == -1);
    tn += (hk == -1 && yk == -1);
    fn += (hk == -1 && yk == 1);
    tp += (hk == 1 && yk == 1);
    if(yk != hk && (i < posImNodes.size())) {
      fprintf(f, "%s\n", posImNodes[i].fileName.c_str());
    }
  }
  fclose(f);

  printf("fp: %d tn: %d fn: %d tp: %d\n", fp, tn, fn, tp);
  double fppw = fp * 1.0 / (tn + fp);
  double miss_rate = fn * 1.0 / (fn + tp);
  printf("fppw: %lf mr: %lf\n", fppw, miss_rate);

  puts("done.");

  return 0;
}
