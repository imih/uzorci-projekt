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
using std::pair;

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

vector<pair<char, int> > allBlocks;
double getVal(const pair<char, int>& a) {
  if(a.first == 't')
    return tvip[a.second];
  else return hvip[a.second];
}

bool cmpAllBlocks(const pair<char, int>& a, const pair<char, int>& b) {
  double v1 = getVal(a);
  double v2 = getVal(b);
  return v1 >= v2;
}

vector<float> getFeats(vector<TextBlock>& t, vector<HOGBlock>& h, int block) {
  //TODO
  vector<float> ret;
  return ret;
}

const int kBlkFactors = 16;

vector<int> sample_ids; 
int last;
void splitSample(Mat& trainData, Mat& trainRes, Mat& valData, Mat& valRes, int block, 
    int k, vector<vector<TextBlock> >& posTex, vector<vector<TextBlock> >& negTex, 
    vector<vector<HOGBlock> >& posHog, vector<vector<HOGBlock> >& negHog) {
  if(!k) last = 0;

  int N = (int) sample_ids.size();
  int Nval = N / 10 + (k + 1 <= N % 10);
  int Ntr = N - Nval;
  int features = 0;
  for(int i = 0; i < block; ++i) {
    features += (allBlocks[i].first == 't' ? posTex[0][0].f.n : posHog[0][0].f.n);
  }

  valRes = Mat(Nval, 1, CV_32F);
  trainRes = Mat(Ntr, 1, CV_32F);
  trainData = Mat(Ntr, features, CV_32F);
  valData = Mat(Nval, features, CV_32F);

  int val_id = 0;
  int train_id = 0;
  for(int i = 0; i < N; ++i) {
    int cur_id = sample_ids[i];
    int& cur_val = train_id;
    Mat& curResp = trainRes;
    Mat& curData = trainData;

    if(i >= last && (k < last + Nval)) {
      //update validation set
      cur_val = val_id;
      curResp = valRes;
      curData = valData;
    }

    vector<float> allFeatures;
    if(cur_id >= 0) {
      curResp.at<float>(cur_val, 0) = 1;
      allFeatures = getFeats(posTex[cur_id], posHog[cur_id], block);
    } else {
      curResp.at<float>(cur_val, 0) = 0;
      cur_id = - cur_id - 1;
      allFeatures = getFeats(negTex[cur_id], negHog[cur_id], block);
    }

    assert((int) allFeatures.size() == features);
    //update cur_val row  of curData with features allFeatures
    for(int i = 0; i < features; ++i) {
      curData.at<float>(cur_val, i) = allFeatures[i];
    }
    cur_val++;
  }

  last += Nval;
}

float errCnt(Mat& h, Mat& y) {
  //jednostupcane matrice TODO
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
  //sort every row by vip score
  for(int i = 0; i < pnt; ++i)
    sort(posTex[i].begin(), posTex[i].end(), cmpTexBlock);
  for(int i = 0; i < nnt; ++i) 
    sort(negTex[i].begin(), negTex[i].end(), cmpTexBlock);
  for(int i = 0; i < pnh; ++i)
    sort(posHog[i].begin(), posHog[i].end(), cmpHogBlock);
  for(int i = 0; i < nnh; ++i)
    sort(negHog[i].begin(), negHog[i].end(), cmpHogBlock);

  for(int i = 0; i < posTex[0].size(); ++i) 
    allBlocks.push_back(make_pair('t', i));
  for(int i = 0; i < posHog[0].size(); ++i)
    allBlocks.push_back(make_pair('h', i));
  sort(allBlocks.begin(), allBlocks.end(), cmpAllBlocks);

  //randomize 
  std::srand((unsigned) time(NULL));
  for(int i = 0; i < (int) posTex.size(); ++i) {
    sample_ids.push_back(i);
  }

  for(int i = 0; i < (int) negTex.size(); ++i) {
    sample_ids.push_back(-i - 1);
  }

  random_shuffle(sample_ids.begin(), sample_ids.end());

  CvSVMParams svmparams;
  svmparams.svm_type = CvSVM::C_SVC;
  svmparams.kernel_type = CvSVM::POLY;
  svmparams.degree = 2;
  svmparams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  CvSVM svm;

  int posBlockSizes[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  vector<double> avgScore(8, 0);

  for(int k = 0; k < 10; ++k) {
    //chose subset of blocks you want to have in the 1st stage using 
    //10-fold cross validation  
    //ajmo prvo bez pls (ostatak TODO )
    for(int i = 0; i < 8; ++i) {
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
  chosenT.clear();
  chosenH.clear();
  for(int i = 0; i < bestBlock; ++i) {
    if(allBlocks[i].first == 'h') {
      chosenH.insert(allBlocks[i].second);
    } else {
      chosenT.insert(allBlocks[i].second);
    }
  }

  return;
}
