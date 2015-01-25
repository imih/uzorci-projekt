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

const int debugFeaturesNo = 200000;
const bool readFeatFromFile = false;
const bool writeFeatToFile = false;
const int maxlen = 2048;

// window size is fixed: 64 x 128 
const int colWin = 64;
const int rowWin = 128;
const int negSampleSize = min(debugFeaturesNo, 10000);

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


void train1();
void train2();
int main(int argc, char** argv) { //have posImNodes and negImNodes now
  puts("Getting features...pos...");
  vector<vector<TextBlock> > perBlockPosTex;
  vector<vector<HOGBlock> > perBlockPosHog;
  vector<vector<TextBlock> > perBlockNegTex; 
  vector<vector<HOGBlock> > perBlockNegHog;
  clock_t begin = clock();

  getTrainingSet();
  for(int im = 0; im < min(debugFeaturesNo, (int) posImNodes.size()); ++im) {
    Mat image = cv::imread(posImNodes[im].fileName, 1);
    Mat curWin = Mat(image, Rect(16, 16, colWin, rowWin)); //BGR
    assert(curWin.rows ==  rowWin);
    assert(curWin.cols ==  colWin);
    perBlockPosTex.push_back(getTextFeatures(curWin));
    perBlockPosHog.push_back(getHOGFeatures(curWin)); 
    clock_t endPos = clock();
    printf("%d/%d t:%0.3lfs\n", im + 1, (int) posImNodes.size(),
        double(endPos - begin) / CLOCKS_PER_SEC);
  }

  puts("loading neg samples...");
  int w = 0;
  vector<cv::Mat> rest;
  for(int im = 0; (im < (int) negImNodes.size()); ++im) {
    Mat image = cv::imread(negImNodes[im].fileName, 1); //BGR
    for(int i = 16; (i + rowWin <= image.rows - 16); i += 8) 
      for(int j = 16; (j + colWin <= image.cols - 16); j += 8) {
        w++;
        Mat curWin = Mat(image, Rect(j, i, colWin, rowWin));
        assert(curWin.rows ==  rowWin);
        assert(curWin.cols ==  colWin);
        rest.push_back(curWin);
        clock_t endPos = clock();
        printf("%d/oo t:%0.3lfs\n", w, negSampleSize, 
            double(endPos - begin) / CLOCKS_PER_SEC);
      } 
  }

  std::srand((unsigned) time(NULL));
  random_shuffle(rest.begin(), rest.end());
  rest.resize(10000); //potrebno? 
  char faza = argv[argc - 1][0];
  vector<vector<TextBlock> > negTexRest;
  vector<vector<HOGBlock> > negHogRest;
  for(int i = 0; i < (int) rest.size(); ++i) {
    if(i < 5000) {
      perBlockNegTex.push_back(getTextFeatures(rest[i]));
      perBlockNegHog.push_back(getHOGFeatures(rest[i]));
    } else if(i >= 5000 && faza == '2') {
      negTexRest.push_back(getTextFeatures(rest[i]));
      negHogRest.push_back(getHOGFeatures(rest[i]));
    }
  }
  rest.clear();
  puts("...done loading  features");
  clock_t endPos = clock();

  if(argv[argc - 1][0] == '2') {
    puts("Performing full pls analysis...\n");

    plsFull(perBlockPosTex, perBlockNegTex, perBlockPosHog, 
        perBlockNegHog, negTexRest, negHogRest);

    /*
       CvSVM SVM;
       SVM.train_auto(mpos + mneg,  ypos + yneg, Mat(), Mat(), params);
       - write ALL parameters to file
       */
  } else if(argv[argc - 1][0] == '1') {
    puts("Performing per block analysis...\n"); 
    // za svaki blok napravi pls i filtriraj koje blokove neces koristiti u stage 1 
    plsPerBlock(perBlockPosTex, perBlockNegTex, perBlockPosHog, perBlockNegHog);
    //TODO
  }

  return 0;
}
