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

const int maxlen = 800000;
char t[maxlen];

template<typename T> 
void readFile(vector<vector<T> >& v, string fileName) {
  ifstream file(fileName.c_str());
  while(file.getline(t, maxlen)) {
    //for each sample:
    int n = strlen(t);
    vector<string> blocks;
    boost::algorithm::split(blocks, t, boost::is_any_of("\t"));
    vector<T> sample;
    int b = (int) blocks.size();
    for(int i = 0; i < b; ++i) {
      sample.push_back(T(blocks[i], i));
    }
    v.push_back(sample);
  }
}

void readTex(vector<vector<TextBlock> >& v, bool ret) {
  readFile(v, "../dataset/tex." + boost::lexical_cast<string>(ret));
}

void readHog(vector<vector<HOGBlock> >& v, bool ret) {
  readFile(v, "../dataset/hog." + boost::lexical_cast<string>(ret));
}

//assuming v[i] is sorted by id
template <typename T>
void writeFile(vector<vector<T> >& v, string fileName) {
  int samples = (int) v.size();
  assert(samples > 0);
  int blocks = (int) v[0].size();

  ofstream file(fileName.c_str());
  for(int i = 0; i < samples; ++i) {
    vector<string> feats;
    for(int j = 0; j < blocks; ++j) {
      vector<string> f;
      for(int k = 0; k < v[i][j].f.n; ++k) {
        f.push_back((boost::format("%.10lf") % v[i][j].f.GetElement(k)).str());
      }
      feats.push_back(join(f, " "));
    }
    file << join(feats, "\t") << std::endl;
  }

  file.close();
}

void writeTex(vector<vector<TextBlock> >& v, bool ret) {
  writeFile(v, "../dataset/tex." + boost::lexical_cast<string>(ret));
}

void writeHog(vector<vector<HOGBlock> >& v, bool ret) {
  writeFile(v, "../dataset/hog." + boost::lexical_cast<string>(ret));
}
