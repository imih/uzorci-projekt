#include <fstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include "texture/TextBlock.h"
#include "hogcolor/HOGBlock.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using texture::TextBlock;
using hog::HOGBlock;
using boost::algorithm::join;

void readTex(vector<vector<TextBlock> >& v, bool ret) {
  //ifstream file("../dataset/tex." + boost::lexical_cast<string>(ret));
  //TODO
}

void readHog(vector<vector<HOGBlock> >& v, bool ret) {
  //ifstream file("../dataset/hog." + boost::lexical_cast<string>(ret));
  //TODO
}

//assuming v[i] is sorted by id
template <typename T>
void writeFile(vector<vector<T> >& v, string fileName) {
  int samples = (int) v.size();
  int blocks = (int) v[0].size();

  ofstream file(fileName.c_str());
  vector<string> blocksS;
  for(int i = 0; i < samples; ++i) {
    vector<string> feats;
    for(int j = 0; j < blocks; ++j) {
      vector<string> f;
      for(int k = 0; k < v[i][j].f.n; ++k) {
        f.push_back((boost::format("%.10lf") % v[i][j].f.GetElement(k)).str());
      }
      feats.push_back(join(f, " "));
    }
    blocksS.push_back(join(feats, "\t"));
  }

  file << join(blocksS, "\n");
  file.close();
}

void writeTex(vector<vector<TextBlock> >& v, bool ret) {
  writeFile(v, "../dataset/tex." + boost::lexical_cast<string>(ret));
}

void writeHog(vector<vector<HOGBlock> >& v, bool ret) {
  writeFile(v, "../dataset/hog." + boost::lexical_cast<string>(ret));
}
