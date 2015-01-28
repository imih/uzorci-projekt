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
        CV_TERMCRIT_EPS, 10000, 10e-6));
};

double getVip(Model& model, int i) { 
  double ret1 = 0, ret2 = 0;
  Matrix<float>* W = model.GetWMatrix();
  Vector<float>* b = model.GetbVector();
  int factors = model.nfactors;

  for(int k = 0; k < factors; ++k) {
    double bk = (double) (*b)[k];
    double wkj = (double) W -> GetElement(k, i);
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
  valData = Mat(Nval, features, CV_32F);

  trainRes = Mat(Ntr, 1, CV_32F);
  valRes = Mat(Nval, 1, CV_32F);
  int blocks = (int) posHog[0].size() + (int) posTex[0].size();

  //jednak broj pozitivnih primjera
  for(int i = 0; i < posTex.size(); ++i) {
    trainRes.at<float>(i, 0) = 1;
    valRes.at<float>(i, 0) = 1;
    vector<float> allFeatures = getFeats(posTex[i], posHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      trainData.at<float>(i, j) = allFeatures[j];
      valData.at<float>(i, j) = allFeatures[j];
    }
  }

  int shiftTrain = (int) posTex.size();
  int shiftVal = (int) posTex.size();
  for(int i = 0; i < negTex.size(); ++i) {
    trainRes.at<float>(shiftTrain + i, 0) = -1;
    vector<float> allFeatures = getFeats(negTex[i], negHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      trainData.at<float>(shiftTrain + i, j) = allFeatures[j];
    }
  }
  shiftTrain += (int) negTex.size();

  int jTrain = 0, jVal = 0;
  for(int i = 0; i < (int) taken.size(); ++i) {
    if(taken[i])  
      trainRes.at<float>(shiftTrain + jTrain, 0) = -1;
    else valRes.at<float>(shiftVal + jVal, 0) = -1;
    vector<float> allFeatures = getFeats(restTex[i], restHog[i], blocks, false);
    for(int j = 0; j < (int) allFeatures.size(); ++j) {
      if(taken[i]) {
        trainData.at<float>(shiftTrain + jTrain, j) = allFeatures[j];
      } else {
        assert(shiftVal + jVal <  Nval);
        valData.at<float>(shiftVal + jVal, j) = allFeatures[j];
      }
    }
    if(taken[i])
      jTrain++;
    else 
      jVal++;
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
}

double errCnt(Mat& h, Mat& y) {
  int n = h.rows;
  double ret = 0;
  int fp = 0, tn = 0, fn = 0, tp = 0;
  int falses = 0;
  for(int i = 0; i < n; ++i) {
    int yk = (int) y.at<float>(i, 0);
    int hk = (int) h.at<float>(i, 0);
    fp += (hk == 1 && yk == -1);
    tn += (hk == -1 && yk == -1);
    fn += (hk == -1 && yk == 1);
    tp += (hk == 1 && yk == 1);
  }
  printf("fp: %d tn: %d fn: %d tp: %d\n", fp, tn, fn, tp);
  double fppw =  fp  * 1.0 / (tn + fp);
  double miss_rate = fn * 1.0 / (fn + tp);
  printf("fppw: %lf mr: %lf\n", fppw, miss_rate);


  double loss01 = fp + fn;
  printf("L01: %lf\n", loss01 / n);
  return (double) loss01 / n;
} 

void plsPerBlock(vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, 
    vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog) {

  puts("getting representatory features for blocks...");
  int blocks = (int) posTex[0].size() + (int) posHog[0].size();

  int pnt = posTex.size();
  int nnt = negTex.size();

  Matrix<float> posData(pnt, blocks);
  Matrix<float> negData(nnt, blocks);

  Model model;
  Matrix<float> *mPos, *mNeg;

  int tblocks = (int) posTex[0].size();
  assert(tblocks == (int) negTex[0].size());

  int mt = posTex[0][0].f.n;
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

    model.CreatePLSModel(mPos, mNeg, mt);

    int bestFeat = 0;
    double bestVip = getVip(model, 0);
    for(int k = 0; k < mt; ++k) {
      double curVip = getVip(model, k);
      if(bestVip < curVip + eps) {
        bestFeat = k;
        bestVip = curVip;
      }
    }

    for(int j = 0; j < pnt; ++j)
      posData.SetValue(j, i,  posTex[j][i].f.GetElement(bestFeat));
    for(int j = 0; j < nnt; ++j)
      negData.SetValue(j, i, negTex[j][i].f.GetElement(bestFeat));

    delete mPos;
    delete mNeg;
  }

  int shift = tblocks;

  int mh = (int) posHog[0][0].f.n;
  int pnh = (int) posHog.size();
  int nnh = (int) negHog.size();

  int hblocks = (int) posHog[0].size();
  assert(hblocks == (int) negHog[0].size());

  for(int i = 0; i < hblocks; ++i) {
    mPos = new Matrix<float>(pnh, mh);
    for(int j = 0; j < pnh; ++j) {
      mPos->SetRow(&posHog[j][i].f, j);
    }

    mNeg = new Matrix<float>(nnh, mh);
    for(int j = 0; j < nnh; ++j) {
      mNeg->SetRow(&negHog[j][i].f, j);
    }

    model.CreatePLSModel(mPos, mNeg, mh);
    int bestFeat = 0;
    double bestVip = getVip(model, 0);
    for(int k = 0; k < mt; ++k) {
      double curVip = getVip(model, k);
      if(bestVip < curVip + eps) {
        bestFeat = k;
        bestVip = curVip;
      }
    }

    for(int j = 0; j < pnh; ++j)
      posData.SetValue(j, i + shift, posHog[j][i].f.GetElement(bestFeat));
    for(int j = 0; j < nnh; ++j)
      negData.SetValue(j, i + shift, negHog[j][i].f.GetElement(bestFeat));

    delete mPos;
    delete mNeg;
  }

  model.CreatePLSModel(&posData, &negData,  kBlkFactors);
  vector<double> vip(blocks, 0);
  for(int i = 0; i < blocks; ++i)
    vip[i] = getVip(model, i);

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
      double v1 = a.first == 't' ? vip[a.second] : vip[a.second + shift];
      double v2 = b.first == 't' ? vip[b.second] : vip[b.second + shift];
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
    17, 18, 19, 20, 21, 22, 23, 24, 35, 26, 27, 28, 29, 30, 31, 32, 40};	
  vector<double> avgScore(33, 0);
  FILE* f = fopen("blocks_scores", "r");

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
      svm.train_auto(trainData, trainRes, Mat(), Mat(), svmparams, 10,
          CvSVM::get_default_grid(CvSVM::C), CvSVM::get_default_grid(CvSVM::GAMMA),
          CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU),
          CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
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

void trainPS(Model& m, CvSVM& svm, Mat& trainData, Mat& trainRes, int factors,
    CvSVMParams& params, bool train_auto = false) {
  Matrix<float>* mTrain = ConvertMatMatrix(trainData);
  Vector<float>* mTrainRes= ConvertMatVector(trainRes);

  m.CreatePLSModel(mTrain, mTrainRes, factors);

  Matrix<float>* plsmTrain = m.ProjectFeatureMatrix(mTrain);
  Mat* newTrainData = ConvertMatrixMat(plsmTrain);

  if(train_auto) {
    auto gridP = CvSVM::get_default_grid(CvSVM::P);
    gridP.step = 0;
    auto gridNu = CvSVM::get_default_grid(CvSVM::NU);
    gridNu.step = 0;
    auto gridDeg = CvSVM::get_default_grid(CvSVM::DEGREE);
    gridDeg.step = 0;
    gridDeg.min_val = 3;
    gridDeg.max_val = 3;
    svm.train_auto(*newTrainData, trainRes, Mat(), Mat(), params, 10, 
        CvSVM::get_default_grid(CvSVM::C), CvSVM::get_default_grid(CvSVM::GAMMA),
        gridP, gridNu,
        CvSVM::get_default_grid(CvSVM::COEF), gridDeg,
        true);
    params = svm.get_params();
  } else {
    svm.train(*newTrainData, trainRes, Mat(), Mat(), params);
  }

  delete mTrain;
  delete mTrainRes;
  delete plsmTrain;
  delete newTrainData;
}


double evaluate(Model& m, CvSVM& svm, Mat& data, Mat& res, 
    vector<bool>* taken = NULL) {
  Matrix<float>* mValid = ConvertMatMatrix(data);
  Matrix<float>* plsmValid = m.ProjectFeatureMatrix(mValid);
  Mat* newValData = ConvertMatrixMat(plsmValid);
  cv::Mat valH = Mat(newValData->rows, 1, CV_32F);
  for(int j = 0; j < newValData->rows; ++j) {		
    valH.at<float>(j, 0) = svm.predict(newValData->row(j), false);
  }
  double err = errCnt(valH, res);

  if(taken != NULL) {
    int fp = 0;
    int j = (int) taken->size() - 1;
    for(int i = res.rows - 1; i >= 0; --i) {
      while(j >= 0 && (*taken)[j]) j--;
      if(j < 0) break;
      if((int) res.at<float>(i, 0) != (int) valH.at<float>(i, 0)) {
        (*taken)[j] = true;
        j--;
        fp++;
      }
    }

    printf("fp: %d\n", fp);
  }

  delete mValid;
  delete plsmValid;
  delete newValData;
  valH.release();
  printf("%lf\n", err);
  return err;
}


void plsFull(vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog, vector<vector<TextBlock> >& restTex,
    vector<vector<HOGBlock> >&restHog) {
  bool trainPLSdimension = true;
  Mat trainData, valData;
  Mat trainRes, valRes;
  CvSVMParams svmparams = getSVMParams();
  CvSVM svm;
  Model model;
  int kFactors = 35;

  if(trainPLSdimension) {
    //randomize 
    std::srand((unsigned) time(NULL));
    sample_ids.clear();
    for(int i = 0; i < (int) posTex.size(); ++i) 
      sample_ids.push_back(i);
    for(int i = 0; i < (int) negTex.size(); ++i) 
      sample_ids.push_back(-i - 1);
    random_shuffle(sample_ids.begin(), sample_ids.end());

    int blockNo = (int) posTex[0].size() + (int) posHog[0].size();
    vector<double> fScores(35, 0.f);

    for(int k = 0;  k < 5; ++k) {
      splitSample(trainData, trainRes, valData, valRes, blockNo, k, posTex, negTex, posHog,
          negHog, false);
      for(int i = 35; i >= 1; --i) {
        printf("k: %d i: %d\n", k, i);
        puts("training...");
        trainPS(model, svm, trainData, trainRes, i, svmparams, false);
        puts("eval...");
        fScores[i] += evaluate(model, svm, valData, valRes);
      }
      trainData.release();
      trainRes.release();
      valData.release();
      valRes.release();
    }

    for(int i = 1; i <= 35; ++i) {
      if(fScores[i] < fScores[kFactors])
        kFactors = i;
      printf("%d: %lf\n", i, fScores[i] / 5);
    }
    printf("Choosing %d dimensions.\n", kFactors);
    exit(0);
  }

  vector<bool> taken((int) restTex.size(), false);

  int maxIt = 10;
  for(int it = 1; it <= maxIt; ++it) {
    printf("it: %d\n", it);
    blocksToFeatures(trainData, trainRes, valData, valRes, posTex, negTex, 
        posHog, negHog, restTex, restHog, taken);
    trainPS(model, svm, trainData, trainRes, kFactors, svmparams, false);
    puts("saving pls model...");
    model.SaveModel("./plsModel2");
    puts("saving svm model...");
    svm.save("./svmModel2.xml");
    puts("eval...");
    evaluate(model, svm, valData, valRes, &taken);

    trainData.release();
    trainRes.release();
    valData.release();
    valRes.release();
  }
}

