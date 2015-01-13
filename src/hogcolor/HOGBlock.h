#pragma once

#include "../pls/maths.h"

namespace hog {

  struct HOGBlock {
    Vector<float> f;
    int block_id;
    HOGBlock(Vector<float> &f_, int bl_id) {
      f = f_;
      block_id = bl_id;
    }
};
}
