#pragma once

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>

namespace texture {
  const int kTexBlockSize = 16;
  const int kColTexBinSize = 16;
  const int kColMax = 255;

  struct Haralick;
  struct TextBlock {
    //contains all textural features for one block accross 3 color channels
    // for one co-occ type
    TextBlock(int);
    std::vector<double> getFeatures();
    void addChannel(int, cv::Mat&);

    private:
    Haralick calcHaralick(cv::Mat&, int) const;
    void createFeatures(); 
    
    std::vector<double> f; //Haralick features
    std::vector<Haralick> texF;
    int coOccType;
  };

  struct Haralick {
    Haralick(cv::Mat&);
    std::vector<double> getFeatures();

    private:
    std::vector<double> f;

  };
}
