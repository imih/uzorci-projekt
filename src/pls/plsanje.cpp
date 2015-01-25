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
using std::pair; const double eps = 10e-6; 
CvSVMParams getSVMParams(){
  return CvSVMParams(CvSVM::C_SVC, CvSVM::POLY, 2, 1, 0, 1, 0, 0, NULL, cvTermCriteria(
        CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1001, FLT_EPSILON));
};

double getVip(Model& model) { 
  double ret1 = 0, ret2 = 0;
  Matrix<float>* W = model.GetWMatrix();
  Vector<float>* b = model.GetbVector();
  int factors = model.nfactors;

  for(int k = 0; k < factors; ++k) {
    double bk = (double) (*b)[k];
    double wkj = (double) W -> GetElement(k, 0);
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


void blocksToFeatures(Mat& trainData, Mat& trainRes, Mat& valData, Mat& valRes,
    vector<vector<TextBlock> >&posTex, vector<vector<TextBlock> >&negTex,
    vector<vector<HOGBlock> >& posHog, vector<vector<HOGBlock> >& negHog,
    vector<vector<TextBlock> >& restTex, vector<vector<HOGBlock> >& restHog,
    vector<bool> taken) {
  int features = (posTex[0].size() * posTex[0][0].f.n + posHog[0].size() * posHog[0][0].f.n);
  assert(posTex[0][0].f.n > 0);
  assert(posHog[0][0].f.n > 0);
  printf("features: %d\n", features);
  int notTaken = 0;
  for(int i = 0; i < (int) taken.size(); ++i)
    notTaken += (taken[i] == false);
  int Ntr = posTex.size() + negTex.size() + (int) taken.size() - notTaken;
  int Nval =  posTex.size() + notTaken;

  trainData = Mat(Ntr, features, CV_32F);
  trainRes = Mat(Ntr, 1, CV_32F);
  valData = Mat(Nval, features, CV_32F);
  valRes = Mat(Nval, 1, CV_32F);
  int blocks = (int) posHog[0].size() + (int) posTex[0].size();

  for(int i = 0; i < posTex.size(); ++i) {
    trainRes.at<float>(i, 0) = 1;
    valRes.at<float>(i, 0) = 1;
    vector<float> allFeatures = getFeats(posTex[i], posHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      trainData.at<float>(i, j) = allFeatures[j];
      valData.at<float>(i, j) = allFeatures[j];
    }
  }

  int shift1 = posTex.size(), shift2 = shift1;
  for(int i = 0; i < negTex.size(); ++i) {
    trainRes.at<float>(shift1 + i, 0) = -1;
    vector<float> allFeatures = getFeats(negTex[i], negHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      trainData.at<float>(shift1 + i, j) = allFeatures[j];
    }
  }
  shift1 += negTex.size();
  for(int i = 0; i < (int) taken.size(); ++i) {
    if(taken[i])
      trainRes.at<float>(shift1 + i, 0) = -1;
    else valRes.at<float>(shift2 + i, 0) = -1;
    vector<float> allFeatures = getFeats(restTex[i], restHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      if(taken[i])
      trainData.at<float>(shift1 + i, j) = allFeatures[j];
      else valData.at<float>(shift2 + i, j) = allFeatures[j];
    }
  }
}

vector<int> sample_ids; 
void splitSample(Mat& trainData, Mat& trainRes, Mat& valData, Mat& valRes, int block, 
    int k, vector<vector<TextBlock> >& posTex, vector<vector<TextBlock> >& negTex, 
    vector<vector<HOGBlock> >& posHog, vector<vector<HOGBlock> >& negHog, 
    bool order = false) {
  int N = (int) sample_ids.size();
  int last = (N / 10) * k + min(k, N % 10);
  int Nval = N / 10 + (k + 1 <= N % 10);
  assert(Nval > 0);
  int Ntr = N - Nval;
  int features = 0;
  if(allBlocks.size()) {
    for(int i = 0; i < block; ++i) 
      features += (allBlocks[i].first == 't' ? posTex[0][0].f.n : posHog[0][0].f.n);
  } else 
    features = (posTex[0].size() * posTex[0][0].f.n + posHog[0].size() * posHog[0][0].f.n);

  assert(posTex[0][0].f.n > 0);
  assert(posHog[0][0].f.n > 0);
  printf("features: %d\n", features);

  trainRes = Mat(Ntr, 1, CV_32F);
  valRes = Mat(Nval, 1, CV_32F);
  trainData = Mat(Ntr, features, CV_32F);
  valData = Mat(Nval, features, CV_32F);

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
  int falses = 0;
  for(int i = 0; i < n; ++i) {
    if((int) y.at<float>(i, 0) != h.at<float>(i, 0))
      falses++;
    double delta =  1. - (double) y.at<float>(i, 0) * h.at<float>(i, 0);
    if(delta >= 10e-6)
      ret += delta;
  }
  printf("%d\n", falses);
  ret /= n;

  return ret;
}

void plsPerBlock(vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, 
    vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog) {
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

    model.CreatePLSModel(mPos, mNeg , kBlkFactors);
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
  sample_ids.clear();
  for(int i = 0; i < (int) posTex.size(); ++i) {
    sample_ids.push_back(i);
  }

  for(int i = 0; i < (int) negTex.size(); ++i) {
    sample_ids.push_back(-i - 1);
  }

  random_shuffle(sample_ids.begin(), sample_ids.end());

  CvSVMParams svmparams = getSVMParams();
  CvSVM svm;

  int posBlockSizes[33] = {1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 40};	
  vector<double> avgScore(33, 0);
  FILE* f = fopen("blocks_scores", "a");

  Mat trainData, trainRes, valData, valRes, valH; 
  puts("performing 10-fold cross validation for stage 1");
  for(int k = 0; k < 10; ++k) {
    //chose subset of blocks you want to have in the 1st stage using 
    //10-fold cross validation  
    // ne treba pls
    for(int i = 0; i < 33; ++i) {
      printf("...%d %d\n", i, k);
      puts("splitting..");
      splitSample(trainData, trainRes, valData, valRes, posBlockSizes[i], k,
          posTex, negTex, posHog, negHog, true);
      puts("training...");
      svm.train(trainData, trainRes, Mat(), Mat(), svmparams);
      valH = Mat(valData.rows, 1, CV_32F);
      for(int j = 0; j < valData.rows; ++j) {		
        valH.at<float>(j, 0) = svm.predict(valData.row(j), true);
      }
      puts("predicted.");
      double err = errCnt(valH, valRes);
      avgScore[i] += err;
    }
  }

  for(int i =0; i < 33; ++i) {
    printf("block %d, score: %lf\n", posBlockSizes[i], avgScore[i]);
    fprintf(f, "block %d, score: %lf\n", posBlockSizes[i], avgScore[i]);
  }

  int bestBlock = posBlockSizes[min_element(avgScore.begin(), avgScore.end()) 
    - avgScore.begin()];
  set<int> chosenT;
  set<int> chosenH;
  for(int i = 0; i < bestBlock; ++i) {
    if(allBlocks[i].first == 'h') {
      chosenH.insert(allBlocks[i].second);
    } else {
      chosenT.insert(allBlocks[i].second);
    }
  }
  printf("block len: %d\n", (int) chosenT.size() + (int) chosenH.size());
  fprintf(f, "block len: %d\n", (int) chosenT.size() + (int) chosenH.size());
  puts("texChosen: ");
  for(int ch1: chosenT) {
    printf("%d " ,ch1);
    fprintf(f, "%d " ,ch1);
  }
  printf("\n");
  fprintf(f, "\n");

  puts("hogChosen: ");
  for(int ch1 : chosenH) {
    printf("%d ", ch1);
    fprintf(f, "%d ", ch1);
  }
  printf("\n");
  fprintf(f, "\n");
  fclose(f);
  return;
}

void plsFull(vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog, vector<vector<TextBlock> >& restTex,
    vector<vector<HOGBlock> >&restHog) {
  Mat trainData, valData;
  Mat trainRes, valRes;

  int kFactors = 25;
  vector<bool> taken((int) restTex.size(), false);
  int maxIt = 10;
  for(int it = 0; it <= maxIt; ++it) {

    CvSVMParams svmparams = getSVMParams();
    CvSVM svm;

    Model model;

    blocksToFeatures(trainData, trainRes, valData, valRes, posTex, negTex, 
        posHog, negHog, restTex, restHog, taken);

    Matrix<float>* mTrain = ConvertMatMatrix(trainData);
    Vector<float>* mVal= ConvertMatVector(valData);
    model.CreatePLSModel(mTrain, mVal, kFactors);

    Matrix<float>* plsmTrain = model.ProjectFeatureMatrix(mTrain);
    Matrix<float>* mValid = ConvertMatMatrix(valData);
    Matrix<float>* plsmValid = model.ProjectFeatureMatrix(mValid);
    if(it == maxIt) {
      puts("saving pls model...");
      model.SaveModel("./plsModel2");
    } 
    model.ClearPLS();

    Mat* newTrainData = ConvertMatrixMat(plsmTrain);
    puts("training...");
    svm.train_auto(*newTrainData, trainRes, Mat(), Mat(), svmparams);
    if(it == maxIt) {
      puts("saving svm model...");
      svm.save("./svmModel2.xml");
    }

    puts("eval...");
    Mat* newValData = ConvertMatrixMat(plsmValid);
    cv::Mat valH = Mat(newValData->rows, 1, CV_32F);
    for(int j = 0; j < valData.rows; ++j) {		
      valH.at<float>(j, 0) = svm.predict(newValData->row(j), true);
    }
    double err = errCnt(valH, valRes);
    printf("%lf\n", err);

    int j = (int) taken.size() - 1;
    for(int i = newValData -> rows - 1; i >= 0; --i) {
      while(j >= 0 && taken[j]) j--;
      if(j < 0) break;
      if((int) newValData->at<float>(i, 0) != (int) valH.at<float>(i, 0))
        taken[j--] = true;
    }


    delete mTrain;
    delete mVal;
    delete plsmTrain;
    delete plsmValid;
    delete newValData;
    delete newTrainData;
  }

}
