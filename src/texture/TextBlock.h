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

    void addFeatures(cv::Mat&);

    Vector<float> f; //Haralick features
    int blockId;

    private:
    int idx;
    int coOccType;
  };
};
