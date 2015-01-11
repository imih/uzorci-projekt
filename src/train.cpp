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
#include "pls/model.h"
#include "pls/maths.h"

using namespace texture;
using namespace hog;
using namespace std;
using namespace cv;

#define cast boost::lexical_cast
#define TRACE(x) std::cout << #x << " = " << x << std::endl

const int maxlen = 2048;

struct ImNode {
  vector<int> Xmin, Ymin, Xmax, Ymax;
  vector<int> cX, cY;
  string fileName;
  int lenX, lenY, channels;
  Mat image;
  ImNode(string name) {
    Xmin.clear();
    Ymin.clear();
    Xmax.clear();
    Ymax.clear();
    fileName = "../dataset/" + name;
  }

  int windows() {
    return (int) Xmin.size();
  }
};
vector<ImNode> posImNodes, negImNodes;

Mat getWindow(ImNode imnode, int w) {
  return Mat(imnode.image, Rect(imnode.Xmin[w], imnode.Ymin[w],
        imnode.Xmax[w] - imnode.Xmin[w] + 1,
        imnode.Ymax[w] - imnode.Ymin[w] + 1));
}

char temp[maxlen];
void getTrainingSet() {
  ifstream anot("../dataset/Train/annotations.lst");
  int posSamples = 0;

  while(anot.getline(temp, maxlen)) {
    string curPath = "../dataset/" + string(temp);
    ifstream curAnot(curPath.c_str());
    ImNode curNode("");
    while(curAnot.getline(temp, maxlen)) {
      int n = strlen(temp);
      if(n <= 3) continue;
      if(temp[0] == '#') continue;
      vector<string> tokens;
      boost::algorithm::split(tokens, temp, boost::is_any_of(" ,()\"-':"), 
          boost::token_compress_on);
      if(tokens[1] == "filename") {
        curNode = ImNode(tokens[2]);
        continue;
      }

      if(tokens[1] == "size") {
        curNode.lenX = cast<int>(tokens[7]);
        curNode.lenY = cast<int>(tokens[9]);
        curNode.channels = cast<int>(tokens[11]);
        continue;
      }

      if(tokens[0] == "Center" && tokens[1] == "point") {
        int cx = cast<int>(tokens[8]);
        int cy = cast<int>(tokens[9]);
        curNode.cX.push_back(cx);
        curNode.cY.push_back(cy);
        continue;
      }

      if(tokens[0] == "Bounding" && tokens[1] == "box") {
        int xmin = cast<int>(tokens[10]);
        curNode.Xmin.push_back(xmin);
        int ymin = cast<int>(tokens[11]);
        curNode.Ymin.push_back(ymin);
        int xmax = cast<int>(tokens[12]);
        curNode.Xmax.push_back(xmax);
        int ymax = cast<int>(tokens[13]);
        curNode.Ymax.push_back(ymax);
        continue;
      }
    }
    curAnot.close();
    posSamples += (int) curNode.Ymax.size();
    posImNodes.push_back(curNode);
  }
  anot.close();
  printf("Read %d pos window annotations from %d images\n", 
      posSamples, (int) posImNodes.size());
  ifstream negList("../dataset/Train/neg.lst");
  while(negList.getline(temp, maxlen)) {
    ImNode curNode(temp);
    negImNodes.push_back(curNode);
  }
  negList.close();

  printf("Read %d neg image info\n", (int) negImNodes.size());

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

vector<double> getTextFeatures(Mat image) {
  vector<double> tex_features;
  for(int p = 0; p < 4; ++p) { // co-occ type 
    vector<TextBlock> t = getTextBlocks(image, p);
    for(int i = 0; i < (int) t.size(); ++i) {
      vector<double> f = t[i].getFeatures();
      tex_features.insert(tex_features.end(), f.begin(), f.end());
    }
  }
  return tex_features;
}

int main(int argc, char** argv) {
  getTrainingSet();
  //have posImNodes and negImNodes now

  puts("Getting features...pos...");
  //get texture features
  //Matrix<double> *mpos = new Matrix<double>(posWinCnt, /* columns */);
  clock_t begin = clock();
  for(int im = 0; im < (int) posImNodes.size(); ++im) {
    for(int w = 0; w < (int) posImNodes[im].windows(); ++w) {
      Mat curWin = getWindow(posImNodes[im], w);
      imshow("win", curWin);
      vector<double> tex_features = getTextFeatures(curWin);
      vector<Mat> hog_features; 
      calc_features(curWin, hog_features);
    }
    clock_t endPos = clock();
    printf("%0.3lfs\n", double(endPos - begin) / CLOCKS_PER_SEC);
  }
  puts("neg...");
  for(int im = 0; im < (int) negImNodes.size(); ++im) {
    vector<double> tex_features = getTextFeatures(negImNodes[im].image);
    vector<Mat> hog_features;
    calc_features(negImNodes[im].image, hog_features);
  }

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
