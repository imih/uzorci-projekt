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

#include "read.h"

using namespace texture;
using namespace hog;
using namespace std;
using namespace cv;

#define TRACE(x) std::cout << #x << " = " << x << std::endl

const bool readFeatFromFile = false;
const bool writeFeatToFile = true;
const int maxlen = 2048;

// window size is fixed: 64 x 128
const int colWin = 64;
const int rowWin = 128;
const int negSampleSize = 10000;

struct ImNode {
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

int main(int argc, char** argv) {
  //have posImNodes and negImNodes now

  puts("Getting features...pos...");
  vector<vector<TextBlock> > perBlockPosTex;
  vector<vector<HOGBlock> > perBlockPosHog;
  vector<vector<TextBlock> > perBlockNegTex; vector<vector<HOGBlock> > perBlockNegHog;
  clock_t begin = clock();

  if(readFeatFromFile) {
    readTex(perBlockPosTex, 1);
    readHog(perBlockPosHog, 1);
    readTex(perBlockNegTex, 0);
    readHog(perBlockNegHog, 0);
  } else {
    getTrainingSet();
    for(int im = 0; im < (int) posImNodes.size(); ++im) {
      Mat curWin = cv::imread(posImNodes[im].fileName, 1); //BGR
      perBlockPosTex.push_back(getTextFeatures(curWin));
      perBlockPosHog.push_back(getHOGFeatures(curWin)); 
      clock_t endPos = clock();
      printf("%d/%d t:%0.3lfs\n", im + 1, (int) posImNodes.size(),
          double(endPos - begin) / CLOCKS_PER_SEC);
    }

    if(writeFeatToFile) {
      writeTex(perBlockPosTex, 1);
      //writeHog(perBlockPosHog, 1);
    }

    puts("neg...");
    int w = 0;
    for(int im = 0; im < (int) negImNodes.size(); ++im) {
      Mat image = cv::imread(negImNodes[im].fileName, 1); //BGR
      for(int i = 0; i + rowWin <= image.rows; i += rowWin) 
        for(int j = 0; j + colWin <= image.cols; j += colWin) {
          w++;
          // mozemo extraktat vise negativnih primjera! TODO
          Mat curWin = Mat(image, Rect(j, i, colWin, rowWin));
          perBlockNegTex.push_back(getTextFeatures(curWin));
          perBlockNegHog.push_back(getHOGFeatures(curWin));
          clock_t endPos = clock();
          printf("%d/%d t:%0.3lfs\n", w, negSampleSize, 
              double(endPos - begin) / CLOCKS_PER_SEC);
          if(w > negSampleSize) break;
        } 
    }
    if(writeFeatToFile) {
      writeTex(perBlockPosTex, 0);
      writeHog(perBlockPosHog, 0);
    }
  }

  puts("done loading  features\n");
  clock_t endPos = clock();
  printf("%0.3lfs\n", double(endPos - begin) / CLOCKS_PER_SEC);
  waitKey(0);

  // za svaki blok napravi pls i filtriraj koje blokove neces koristiti u stage 1 
  set<int> texChosen, hogChosen;
  puts("Performing per block analysis...\n"); 
  plsPerBlock(perBlockPosTex, perBlockNegTex, texChosen,
      perBlockPosHog, perBlockNegHog, hogChosen);
  
  //cross validate:
  //1) n_factors - stage 1
  //2) n_factors - stage 2 TODO

  /*

     CvSVM SVM;
     SVM.train_auto(mpos + mneg,  ypos + yneg, Mat(), Mat(), params);
  //write ALL parameters to file
  */

  return 0;
}
