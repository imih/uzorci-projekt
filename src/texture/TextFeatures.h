#include <vector>

#include "TextBlock.h"
#pragma once

#include <set>

namespace texture {
  std::vector<texture::TextBlock> getTextBlocks(cv::Mat, std::set<int> = std::set<int>());
};
