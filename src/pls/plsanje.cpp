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

const double eps = 10e-6;

CvSVMParams getSVMParams(){
  return CvSVMParams(CvSVM::C_SVC, CvSVM::POLY, 2, 1, 0, 1, 0, 0, NULL, cvTermCriteria(
        CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON));
};

double getVip(Model& model) { 
  double ret1 = 0, ret2 = 0;
  Matrix<float>* W = model.GetWMatrix();
  Vector<float>* b = model.GetbVector();
  int factors = b->GetNElements();
  assert(factors > 0);

  for(int k = 0; k < factors; ++k) {
    int bk = (double) (*b)[k];
    int wkj = (double) W -> GetElement(k, 0);
    ret1 += bk * bk * wkj * wkj;
    ret2 += bk * bk;
  }

  return sqrt(ret1 * factors / ret2);
}

vector<pair<char, int> > allBlocks;
vector<float> getFeats(vector<TextBlock>& t, vector<HOGBlock>& h, int block, 
    bool order = false) {
  vector<float> ret;
  int t_id = 0, h_id = 0;
  for(int i = 0; i < block; ++i) {
    if(order) {
      pair<char, int> cur = allBlocks[i];
      if(cur.first == 't') {
        for(int j = 0; j < t[cur.second].f.n; ++j) 
          ret.push_back(t[cur.second].f.GetElement(j));
      } else {
        for(int j = 0; j < h[cur.second].f.n; ++j)
          ret.push_back(h[cur.second].f.GetElement(j)); }
    } else {
      if(t_id != (int) t.size()) {
        for(int j = 0; j < t[t_id].f.n; ++j) 
          ret.push_back(t[t_id].f.GetElement(j));
        t_id++;
      } else {
        for(int j = 0; j < h[h_id].f.n; ++j)
          ret.push_back(h[h_id].f.GetElement(j)); 
        h_id++;
      }
    }
  }
  return ret;
}

const int kBlkFactors = 16;

vector<int> sample_ids; 
int last;
void splitSample(Mat& trainData, Mat& trainRes, Mat& valData, Mat& valRes, int block, 
    int k, vector<vector<TextBlock> >& posTex, vector<vector<TextBlock> >& negTex, 
    vector<vector<HOGBlock> >& posHog, vector<vector<HOGBlock> >& negHog, bool order = false) {
  if(!k) last = 0;

  int N = (int) sample_ids.size();
  int Nval = N / 10 + (k + 1 <= N % 10);
  int Ntr = N - Nval;
  int features = 0;
  for(int i = 0; i < block; ++i) {
    features += (allBlocks[i].first == 't' ? posTex[0][0].f.n : posHog[0][0].f.n);
  }

  valRes = Mat(Nval, 1, CV_32F);
  valRes.addref();
  trainRes = Mat(Ntr, 1, CV_32F);
  trainRes.addref();
  trainData = Mat(Ntr, features, CV_32F);
  trainData.addref();
  valData = Mat(Nval, features, CV_32F);
  valData.addref();

  int val_id = 0;
  int train_id = 0;
  for(int i = 0; i < N; ++i) {
    int cur_id = sample_ids[i];
    bool getToVal = ((i >= last) && (i < last + Nval));

    vector<float> allFeatures;
    if(cur_id >= 0) {
      if(!getToVal) {
        trainRes.at<float>(train_id, 0) = 1;
      } else {
        valRes.at<float>(val_id, 0) = 1;
      }
      allFeatures = getFeats(posTex[cur_id], posHog[cur_id], block, order);
    } else {
      if(!getToVal)
        trainRes.at<float>(train_id, 0) = -1;
      else 
        valRes.at<float>(val_id, 0) = -1;
      cur_id = -cur_id - 1;
      allFeatures = getFeats(negTex[cur_id], negHog[cur_id], block, order);
    }

    assert((int) allFeatures.size() == features);
    //update cur_val row  of curData with features allFeatures
    for(int i = 0; i < features; ++i) {
      if(!getToVal)
        trainData.at<float>(train_id, i) = allFeatures[i];
      else valData.at<float>(val_id, i) = allFeatures[i];
    }
    if(!getToVal)
      train_id++;
    else val_id++;
  }

  last += Nval;
}

double errCnt(Mat& h, Mat& y) {
  int n = h.rows;
  double ret = 0;
  for(int i = 0; i < n; ++i) {
    if(fabs(y.at<float>(i, 0) - h.at<float>(i, 0)) > 10e-6)
      ret +=  (y.at<float>(i, 0) - h.at<float>(i, 0)) * 
        log(fabs(y.at<float>(i, 0) - h.at<float>(i, 0)));
  }

  return ret;
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
  puts("getting vip for texture blocks...");
  int mt  = (int) posTex[0][0].f.n;
  int pnt = (int) posTex.size();
  int nnt = (int) negTex.size();

  int tblocks = (int) posTex[0].size();
  assert(tblocks == (int) negTex[0].size());
  vector<double> tvip = vector<double>(tblocks, 0);
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
  }

  puts("getting vip for hog blocks...");
  int mh = (int) posHog[0][0].f.n;
  int pnh = (int) posHog.size();
  int nnh = (int) negHog.size();

  int hblocks = (int) posHog[0].size();
  assert(hblocks == (int) negHog[0].size());
  vector<double> hvip = vector<double>(hblocks, 0);
  for(int i = 0; i < hblocks; ++i) {
    mPos = new Matrix<float>(pnh, mh);
    for(int j = 0; j < pnh; ++j) {
      mPos->SetRow(&posHog[j][i].f, j);
    }

    mNeg = new Matrix<float>(nnh, mh);
    for(int j = 0; j < nnh; ++j) {
      mNeg->SetRow(&negHog[j][i].f, j);
    }

    model.CreatePLSModel(mPos, mNeg , 3);
    hvip[i] = getVip(model);
  }

  //**************************************************************
  //sort all blocks by vip score
  allBlocks.clear();
  for(int i = 0; i < (int) posTex[0].size(); ++i) 
    allBlocks.push_back(make_pair('t', i));
  for(int i = 0; i < (int) posHog[0].size(); ++i)
    allBlocks.push_back(make_pair('h', i));
  sort(allBlocks.begin(), allBlocks.end(), [&](
        const pair<char, int>& a, const pair<char, int>&b) {
      assert(a.first == 't' || a.first == 'h');
      assert(b.first == 't' || b.first == 'h');
      double v1 = a.first == 't' ? tvip[a.second] : hvip[a.second];
      double v2 = b.first == 't' ? tvip[b.second] : hvip[b.second];
      if(fabs(v1 - v2) <= eps) return false;
      return v1 + eps >= v2;
      });
  puts("done sorting\n");

  //randomize 
  std::srand((unsigned) time(NULL));
  for(int i = 0; i < (int) posTex.size(); ++i) {
    sample_ids.push_back(i);
  }

  for(int i = 0; i < (int) negTex.size(); ++i) {
    sample_ids.push_back(-i - 1);
  }

  random_shuffle(sample_ids.begin(), sample_ids.end());

  CvSVMParams svmparams = getSVMParams();
  CvSVM svm;

  int posBlockSizes[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  vector<double> avgScore(8, 0);

  Mat trainData, trainRes, valData, valRes, valH; 
  puts("performing 10-fold cross validation for stage 1");
  for(int k = 0; k < 10; ++k) {
    //chose subset of blocks you want to have in the 1st stage using 
    //10-fold cross validation  
    // ne treba pls
    for(int i = 0; i < 8; ++i) {
      printf("...%d %d\n", i, k);
      printf("splitting..");
      splitSample(trainData, trainRes, valData, valRes, posBlockSizes[i], k,
          posTex, negTex, posHog, negHog, true);
      printf("training...");
      for(int i = 0; i < trainRes.rows; ++i)
        printf("%lf\n", trainRes.at<float>(i, 0));
      svm.train(trainData, trainRes, Mat(), Mat(), svmparams);
      svm.predict(valData, valH);
      printf("predicted.");
      double err = errCnt(valH, valRes);
      avgScore[i] += err;
    }
  }

  int bestBlock = posBlockSizes[min_element(avgScore.begin(), avgScore.end()) 
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

void plsFull(int n_factors_best, vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, set<int>& chosenT, vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog, set<int>& chosenH) {
  //choose n_factors for 2nd (full) stage and the blocks you want to read them on

  //randomize 
  std::srand((unsigned) time(NULL));
  for(int i = 0; i < (int) posTex.size(); ++i) {
    sample_ids.push_back(i);
  }

  for(int i = 0; i < (int) negTex.size(); ++i) {
    sample_ids.push_back(-i - 1);
  }
  random_shuffle(sample_ids.begin(), sample_ids.end());

  CvSVMParams svmparams = getSVMParams();
  CvSVM svm;
  Model model;
  Mat trainData, valData;
  Mat trainRes, valRes;
  Mat valH; 

  int nfactors[13] = {2, 4, 10, 15, 20, 25, 30, 35, 40, 60, 100, 250, 500};
  vector<double> avgScore(13, 0);

  //10-fold cross validation  
  int blocks_no = (int) posTex[0].size() + (int) posHog[0].size();
  for(int k = 0; k < 10; ++k) {
    for(int i = 0; i < 13; ++i) {
      splitSample(trainData, trainRes, valData, valRes, blocks_no, k, posTex, negTex, 
          posHog, negHog);

      Matrix<float>* mTrain;
      ConvertMatMatrix(trainData, mTrain);
      Vector<float>* mVal;
      ConvertMatVector(valData, mVal);
      model.CreatePLSModel(mTrain, mVal, nfactors[i]);

      Matrix<float>* plsmTrain = model.ProjectFeatureMatrix(mTrain);
      Matrix<float>* mValid;
      ConvertMatMatrix(valData, mValid);
      Matrix<float>* plsmValid = model.ProjectFeatureMatrix(mValid);

      ConvertMatrixMat(plsmTrain, &trainData);
      svm.train(trainRes, trainRes, Mat(), Mat(), svmparams);

      ConvertMatrixMat(plsmValid, &valData);
      svm.predict(valData, valH);
      double err = errCnt(valH, valRes);
      avgScore[i] += err;
    }
  }

  n_factors_best = nfactors[min_element(avgScore.begin(), avgScore.end()) - avgScore.begin()];
  //TODO ne znam dobiti samo one blokove koji se koriste  za pls!
}
