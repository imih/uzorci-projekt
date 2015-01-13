#pragma once

#include "../pls/maths.h"

namespace hog {

  struct HOGBlock {
    Vector<float> f;
    int blockId;
    HOGBlock(Vector<float> &f_, int block_id) {
      f = f_;
      blockId = block_id;
    }

};
}
