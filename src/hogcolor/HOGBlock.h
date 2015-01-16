#pragma once

#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "../pls/maths.h"

namespace hog {

  struct HOGBlock {
    Vector<float> f;
    int block_id;
    HOGBlock(Vector<float> &f_, int bl_id) {
      f = f_;
      block_id = bl_id;
    }

    HOGBlock(std::string &s, int bl_id) {
      std::vector<std::string> fs;
      boost::algorithm::split(fs, s, boost::is_any_of(" "));
      int n = (int) fs.size();
      f = Vector<float>(n);
      for(int i = 0; i < n; ++i)
        f.SetElement(i, boost::lexical_cast<float>(fs[i]));
      block_id = bl_id;
    }

    void output() {
      puts("HOG-R");
      for (int i = 0; i < 12; ++i)
	printf("%.3f ", f.GetElement(i));
      puts("");
      puts("HOG-G");
      for (int i = 0; i < 12; ++i)
	printf("%.3f ", f.GetElement(i + 12));
      puts("");
      puts("HOG-B");
      for (int i = 0; i < 12; ++i)
	printf("%.3f ", f.GetElement(i + 24));
      puts("");
      puts("RGB");
      for (int i = 0; i < 3; ++i)
	printf("%.3f ", f.GetElement(i + 36));
      puts("");
      puts("");
    }
  };
}
