#include <opencv2/opencv.hpp>
#include <vector>
#pragma once

#include <vector>
#include <set>

#include <opencv2/opencv.hpp>

#include "HOGBlock.h"

namespace hog { 
  void calc_features(cv::Mat&, std::vector<HOGBlock>&, std::set<int> = std::set<int>());
};

