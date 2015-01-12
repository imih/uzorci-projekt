#pragma once

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>

#include "../pls/maths.h"

namespace texture {
  using std::vector;
  
  const int kColTexBinSize = 16;
  const int kColMax = 255;

  struct TextBlock {
    //contains all textural features for one block accross 3 color channels
    // for one co-occ type
    TextBlock(int, int);
    TextBlock();

    Vector<float> getFeatures();
    void addChannel(int, cv::Mat&);

    int blockId;

    private:
    vector<float> calcHaralick(cv::Mat&, int);
    void createFeatures(); 
    
    vector<float> texF;
    Vector<float> f; //Haralick features
    int coOccType;
  };
};
