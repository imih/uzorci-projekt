#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace texture {
  class TextBlock {
    public:
      TextBlock(cv::Mat occ);
      std::vector<double> toFeatures();

    private:
      std::vector<double> f; //Haralick features
  };
}
