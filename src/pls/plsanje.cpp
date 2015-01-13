#include <cstdio>
#include <algorithm>

#include "math.h"
#include "model.h"
#include "plsanje.h"


using std::vector;
using texture::TextBlock;
using hog::HOGBlock;
using std::random_shuffle;
using std::sort;
using cv::Mat;

double getVip(Model& model) { double ret1 = 0, ret2 = 0;
  Matrix<float>* W = model.GetWMatrix();
  Vector<float>* b = model.GetbVector();
  for(int k = 0; k < model.nfactors; ++k) {
    int bk = (double) (*b)[k];
    int wkj = (double) W -> GetElement(k, 0);
    ret1 += bk * bk * wkj * wkj;
    ret2 += bk * bk;
  }

  return sqrt(ret1 * model.nfeatures  / ret2);
}

vector<double> tvip;
bool cmpTexBlock(const TextBlock& a, const TextBlock& b){
  return tvip[a.block_id] > tvip[b.block_id];
}

vector<double> hvip;
bool cmpHogBlock(const HOGBlock& a, const HOGBlock& b) {
  return hvip[a.block_id] > hvip[b.block_id];
}

const int kBlkFactors = 16;

void splitSample(Mat& trainData, Mat& trainRes, Mat& valData, Mat& valRes, int block, 
    int k, vector<vector<TextBlock> >& posTex, vector<vector<TextBlock> >& negTex, 
    vector<vector<HOGBlock> >& posHog, vector<vector<HOGBlock> >& negHog) {
  //create  validation and training set TODO
}

float errCnt(Mat& h, Mat& y) {
  //TODO
  return 0;
}

void plsPerBlock(vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, 
    set<int>& chosenT,
    vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog, 
    set<int>& chosenH) {
  Model model;
  Matrix<float> *mPos, *mNeg;

  //calc  rank for each block
  int tblocks = (int) posTex[0].size();
  tvip = vector<double>(tblocks, 0);
  int mt  = (int) posTex[0][0].f.n;
  int pnt = (int) posTex.size();
  int nnt = (int) negTex.size();
  for(int i = 0; i < tblocks; ++i) {
    mPos = new Matrix<float>(pnt, mt);
    for(int j = 0; j < pnt; ++j) {
      //i-ti blok u j-tom primjeru
      mPos->SetRow(&posTex[j][i].f, j);
    }

    mNeg = new Matrix<float>(nnt, mt);
    for(int j = 0; j < nnt; ++j) {
      mNeg->SetRow(&negTex[j][i].f, j);
    }

    model.CreatePLSModel(mPos, mNeg, kBlkFactors);
    tvip[i] = getVip(model);
    delete mPos, mNeg;
  }

  int hblocks = (int) posHog[0].size();
  hvip = vector<double>(hblocks, 0);
  int mh = (int) posHog[0][0].f.n;
  int pnh = (int) posHog.size();
  int nnh = (int) negHog.size();
  for(int i = 0; i < hblocks; ++i) {
    mPos = new Matrix<float>(pnh, mh);
    for(int j = 0; j < pnh; ++j) {
      mPos->SetRow(&posHog[j][i].f, j);
    }

    mNeg = new Matrix<float>(nnh, mh);
    for(int j = 0; j < nnh; ++j) {
      mNeg->SetRow(&negHog[j][i].f, j);
    }
    model.CreatePLSModel(mPos, mNeg , kBlkFactors);
    hvip[i] = getVip(model);
    delete mPos, mNeg;
  }

  //**************************************************************
  //chose subset of blocks you want to have in the 1st stage using 
  //10-fold cross validation  
  int pt_n = (int) posTex.size() / 10;
  int nt_n = (int) negTex.size() / 10;
  int ph_n = (int) posHog.size() / 10;
  int nh_n = (int) negHog.size() / 10;

  //sort every row by vip score
  for(int i = 0; i < pnt; ++i) 
    sort(posTex[i].begin(), posTex[i].end(), cmpTexBlock);
  for(int i = 0; i < nnt; ++i) 
    sort(negTex[i].begin(), negTex[i].end(), cmpTexBlock);
  for(int i = 0; i < pnh; ++i) 
    sort(posHog[i].begin(), posHog[i].end(), cmpHogBlock);
  for(int i = 0; i < nnh; ++i)
    sort(negHog[i].begin(), negHog[i].end(), cmpHogBlock);

  /*
     std::srand((unsigned) time(NULL));
     random_shuffle(posTex.begin(), posTex.end());
     random_shuffle(negTex.begin(), negTex.end());
     random_shuffle(posHog.begin(), posHog.end());
     random_shuffle(negHog.begin(), negHog.end());
     */

  CvSVMParams svmparams;
  svmparams.svm_type = CvSVM::C_SVC;
  svmparams.kernel_type = CvSVM::POLY;
  svmparams.degree = 2;
  svmparams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  CvSVM svm;

  int posBlockSizes[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  vector<double> avgScore(8, 0);

  for(int k = 0; k < 10; ++k) {
    //ajmo prvo bez pls (ostatak TODO )
    for(int i = 0; i < 8; ++i) {
      // train and validate TODO
      Mat trainData, trainRes, valData, valRes, valH;
      splitSample(trainData, trainRes, valData, valRes, posBlockSizes[i], k,
          posTex, negTex, posHog, negHog);
      svm.train(trainData, trainRes, Mat(), Mat(), svmparams);
      svm.predict(valData, valH);
      float err = errCnt(valH, valRes);
      avgScore[i] += err;
    }
  }

  int bestBlock = posBlockSizes[max_element(avgScore.begin(), avgScore.end()) 
    - avgScore.begin()];
  int t = 0, h = 0;
  chosenT.clear();
  chosenH.clear();
  for(int i = 0; i < bestBlock; ++i) {
    int tid = posTex[0][t].block_id;
    int hid = posHog[0][h].block_id;
    if(tvip[tid] < hvip[hid]) {
      chosenH.insert(hid);
      h++;
    } else {
      chosenT.insert(tid);
      t++;
    }
  }

  return;
}
