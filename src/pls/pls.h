/*
 * The implementation of the NIPALS algorithm provided in this library
 * is a translation from the MATLAB version of the NIPALS algorithm 
 * written by Dr. Herve Abdi from The University of Texas at Dallas
 * http://www.utdallas.edu/~herve.
 */

#ifndef PLS_H
#define PLS_H
#include "headers.h"
#include "maths.h"

class PLS {
  Vector<double> *Xmean, *Xstd, *Yorig, *b;
  Matrix<double> *T, *P, *W, *Wstar;
  int maxFactors;		// maximum number of factors for this model

  // other variables
  Maths mat;
  Vector<double> *zdataV;		// variable to hold result of zscore


  void normaliz(Vector<double> *vector, Vector<double> *retvector);
  void MultiplyTransposeMatrixbyVector(Matrix<double> *M, Vector<double> *v, Vector<double> *retvector);
  void MultiplyMatrixbyVector(Matrix<double> *M, Vector<double> *v, Vector<double> *retvector);
  double MultiplyVectorTransposedbyVector(Vector<double> *v1, Vector<double> *v2);
  void MultiplyVectorandScalar(Vector<double> *v, double s, Vector<double> *retvector);
  void SubtractFromMatrix(Matrix<double> *M, Vector<double> *t, Vector<double> *p);
  void SubtractFromVector(Vector<double> *v, Vector<double> *t, double c, double bl);
  void CopyVector(Vector<double> *v, Vector<double> *retvector);
  void mean(Matrix<double> *M, Vector<double> *retvector) ;
  void mean(Vector<double> *M, Vector<double> *retvector) ;
  void std(Matrix<double> *M, Vector<double> *mean, Vector<double> *retvector);
  void std(Vector<double> *M, Vector<double> *mean, Vector<double> *retvector);
  void zscore(Matrix<double> *M, Vector<double> *mean, Vector<double> *std);
  void zscore(Vector<double> *M, Vector<double> *mean, Vector<double> *std);
  void ComputeWstar();

  // clear data not used for PLS regression (leave only Bstar, Ymean, Xstd, Xmean)
  void ClearExtraMatricesPLSReg();

  friend class Storage;



  protected:

  // set all matrices initializing with already computed values 
  // Warning: this function COPY all variables
  void InitializePLSModel(Vector<double> *Xmean, Vector<double> *Xstd, Vector<double> *Yorig, Vector<double> *b, Matrix<double> *T, 
      Matrix<double> *P, Matrix<double> *W, Matrix<double> *Wstar);

  // remove matrices not used for projections
  void ClearExtraMatrices();

  // return projection considering n factors
  void Projection(double *feat, double *retproj, int nfactors);

  // execute PLS for maximum number of factors: nfactor
  void runpls(Matrix<double> *X, Vector<double> *Y, int nfactor, char *OutputDir = NULL, double ExplainedX = 1, double ExplainedY = 1);

  // return feature vector running zcore
  void ExecuteZScore(double *feat, double *zscoreResult);

  // set matrices for PLS
  void SetMatrices(Matrix<double> *W, Matrix<double> *Wstar, Matrix<double> *P, Vector<double> *Xmean, Vector<double> *Xstd, Vector<double> *b);


  // clear variables of this class
  void ClearPLS();


  // friend classes
  //friend class Storage;
  friend class Structures;

  // Return matrices and vectors
  Matrix<double> *GetWMatrix() { return W; }
  Matrix<double> *GetTMatrix() { return T; }
  Matrix<double> *GetPMatrix() { return P; }
  Matrix<double> *GetWstar() { return Wstar; }
  Vector<double> *GetbVector() { return b; }
  Vector<double> *GetYVector() { return Yorig; }
  Vector<double> *GetMeanVector() { return Xmean; }
  Vector<double> *GetStdVector() { return Xstd; }
  Vector<double> *GetBstar(int nfactors);

  public:
  PLS();
  ~PLS();
};


#endif
