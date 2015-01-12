#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>

#include <boost/algorithm/string.hpp> 
#include <boost/lexical_cast.hpp>

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

#define cast boost::lexical_cast
#define TRACE(x) std::cout << #x << " = " << x << std::endl

const int maxlen = 2048;

// window size is fixed: 64 x 128
const int colWin = 64;
const int rowWin = 128;

struct ImNode {
  Mat image;
  string fileName;
  ImNode(string name) {
    fileName = name;
  }
};

vector<ImNode> posImNodes, negImNodes;

char temp[maxlen];
void getTrainingSet() {
  ifstream posList("../dataset/train_64x128_H96/pos.lst");
  while(posList.getline(temp, maxlen)) {
    string curPath = "../dataset/" + string(temp);
    posImNodes.push_back(ImNode(curPath));
  }
  posList.close();
  printf("Read %d pos window samples\n", (int) posImNodes.size());

  ifstream negList("../dataset/train_64x128_H96/neg.lst");
  while(negList.getline(temp, maxlen)) {
    string curPath = "../dataset/" + string(temp);
    negImNodes.push_back(ImNode(curPath));
  }
  negList.close();
  printf("Read %d neg window samples\n", (int) negImNodes.size()); 
  puts("Loading images...");
  clock_t begin = clock();
  for(int i = 0; i < (int) posImNodes.size(); ++i) {
    posImNodes[i].image = cv::imread(posImNodes[i].fileName, 1); //BGR
    assert(posImNodes[i].image.rows > 0);
  }
  for(int i = 0; i < (int) negImNodes.size(); ++i) {
    negImNodes[i].image = cv::imread(negImNodes[i].fileName, 1); //BGR
    assert(negImNodes[i].image.rows > 0);
  }
  clock_t end = clock();
  printf("%0.3lfs\n", double(end - begin) / CLOCKS_PER_SEC);
}

vector<TextBlock > getTextFeatures(Mat image) {
  return getTextBlocks(image);
}

vector<HOGBlock> getHOGFeatures(Mat image) {
  vector<HOGBlock> ret;
  calc_features(image, ret);
  return ret;
}

int main(int argc, char** argv) {
  getTrainingSet();
  //have posImNodes and negImNodes now

  puts("Getting features...pos...");
  vector<vector<TextBlock> > perBlockPosTex;
  vector<vector<HOGBlock> > perBlockPosHog;
  clock_t begin = clock();
  for(int im = 0; im < (int) posImNodes.size(); ++im) {
    Mat curWin = posImNodes[im].image;
    vector<TextBlock> tex_features = getTextFeatures(curWin);
    perBlockPosTex.push_back(tex_features);
    vector<HOGBlock> hog_features = getHOGFeatures(curWin);
    perBlockPosHog.push_back(hog_features);
    clock_t endPos = clock();
    printf("%0.3lfs\n", double(endPos - begin) / CLOCKS_PER_SEC);
  }

  puts("neg...");
  vector<vector<TextBlock> > perBlockNegTex;
  vector<vector<HOGBlock> > perBlockNegHog;
  for(int im = 0; im < (int) negImNodes.size(); ++im) 
    for(int i = 0; i + rowWin <= negImNodes[im].image.rows; i += rowWin) 
      for(int j = 0; j + colWin <= negImNodes[im].image.cols; j += colWin) {
        Mat curWin = Mat(posImNodes[im].image, Rect(i, j, rowWin, colWin));
        vector<TextBlock> tex_features = getTextFeatures(curWin);
        perBlockNegTex.push_back(tex_features);
        vector<HOGBlock> hog_features = getHOGFeatures(curWin);
        perBlockNegHog.push_back(hog_features);
        clock_t endPos = clock();
        printf("%0.3lfs\n", double(endPos - begin) / CLOCKS_PER_SEC);
      } 

  puts("done loading  features\n");
  clock_t endPos = clock();
  printf("%0.3lfs\n", double(endPos - begin) / CLOCKS_PER_SEC);

  // za svaki blok napravi pls i filtriraj koje blokove neces koristiti u stage 1 
  set<int> texSkip, hogSkip;
  plsPerBlock(perBlockPosTex, perBlockNegTex, texSkip,
     perBlockPosHog, perBlockNegHog, hogSkip);











  /*
  //neg texture
  Matrix<double> *mneg = 
  //pls on texture features
  //get hog 
  //pls on hog

  //SVM parameters *******************
  CvSVMParams params;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit = cvTermCriteria(CV_TEMCRIT_ITER, 100, 1e-6);

  CvSVM SVM;
  SVM.train(mpos + mneg,  ypos + yneg, Mat(), Mat(), params);
  //write ALL parameters to file
  */

  return 0;
}
