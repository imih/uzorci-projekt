#pragma once

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>

#include "../pls/maths.h"

namespace texture {
  const int kColTexBinSize = 16;
  const int kColMax = 255;

  struct Haralick;
  struct TextBlock {
    //contains all textural features for one block accross 3 color channels
    // for one co-occ type
    TextBlock(int, int);
    TextBlock();

    Vector<float> getFeatures();
    void addChannel(int, cv::Mat);
    int blockId;

    private:
    Haralick calcHaralick(cv::Mat, int) const;
    void createFeatures(); 
    
    Vector<float> f; //Haralick features
    std::vector<Haralick> texF;
    int coOccType;
  };

  struct Haralick {
    Haralick(cv::Mat);
    std::vector<float> getFeatures();

    private:
    std::vector<float> f;
  };
}
