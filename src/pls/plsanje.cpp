#include <cstdio>
#include <algorithm>

#include "math.h"
#include "model.h"
#include "plsanje.h"

using std::vector;
using texture::TextBlock;
using hog::HOGBlock;
using std::random_shuffle;

double getVip(Model& model) {
  double ret1 = 0, ret2 = 0;
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

const int kBlkFactors = 16;
void plsPerBlock(vector<vector<TextBlock> >& posTex, 
    vector<vector<TextBlock> >& negTex, 
    set<int>& skipTex,
    vector<vector<HOGBlock> >& posHog, 
    vector<vector<HOGBlock> >& negHog, 
    set<int>& skipHog) {
  Model model;
  Matrix<float> *mPos, *mNeg;

  //calc  rank for each block
  int tblocks = (int) posTex[0].size();
  vector<double> tvip(tblocks, 0);
  for(int i = 0; i < tblocks; ++i) {
    //TODO
    model.CreatePLSModel(mPos, mNeg, kBlkFactors);
    tvip[i] = getVip(model);
    delete mPos, mNeg;
  }

  int hblocks = (int) posHog[0].size();
  vector<double> hvip(tblocks, 0);
  for(int i = 0; i < hblocks; ++i) {
    //TODO
    model.CreatePLSModel(mPos, mNeg , kBlkFactors);
    hvip[i] = getVip(model);
    delete mPos, mNeg;
  }

  //chose subset of blocks you want to have in the 1st stage using 
  //10-fold cross validation  TODO
  int pt_n = (int) posTex.size() / 10;
  int nt_n = (int) negTex.size() / 10;
  int ph_n = (int) posHog.size() / 10;
  int ng_n = (int) negHog.size() / 10;

  std::srand((unsigned) time(NULL));
  random_shuffle(posTex.begin(), posTex.end());
  random_shuffle(negTex.begin(), negTex.end());
  random_shuffle(posHog.begin(), posHog.end());
  random_shuffle(negHog.begin(), negHog.end());


  //update the sets TODO


  return;
}
