#include <vector>

#include "TextBlock.h"
#pragma once

#include <set>

namespace texture {
  void getTextBlocks(cv::Mat&, std::vector<TextBlock>&, std::set<int> = std::set<int>());
};
