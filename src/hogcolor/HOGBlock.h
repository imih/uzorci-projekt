#pragma once

#include <opencv2/opencv.hpp>

namespace hog {
  struct HOGBlock {
    cv::Mat f;
    int blockId;
    HOGBlock(cv::Mat f_, int block_id) {
      f = f_;
      blockId = block_id;
    }
};
}
