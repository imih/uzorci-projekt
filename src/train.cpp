#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp> 
#include <boost/lexical_cast.hpp>

#include "texture/TextFeatures.h"
#include "hogcolor/HOGAndColorFeatures.h"
#include "pls/model.h"
#include "pls/maths.h"


using namespace texture;
using namespace hog;
using namespace std;

#define cast boost::lexical_cast
#define TRACE(x) std::cout << #x << " = " << x << std::endl

struct ImNode {
  std::vector<int> Xmin, Ymin, Xmax, Ymax;
  std::vector<int> cX, cY;
  string fileName;
  int lenX, lenY, channels;
  ImNode(string name) {
    Xmin.clear();
    Ymin.clear();
    Xmax.clear();
    Ymax.clear();
    fileName = name;
  }

};

void getTrainingSet() {
  puts("Texture features extraction\n");

  vector<ImNode> posImgs;
  FILE *anot = fopen("../dataset/Train/annotations.lst", "r");
  vector<ImNode> imNodes;

  char temp[1024];
  while(fgets(temp, 1024, anot)) {
    printf("%s\n", temp);
    string curPath = "../dataset/" + string(temp);
    FILE *curAnot = fopen(curPath.c_str(), "r");
    ImNode curNode("");
    while(fgets(temp, 1024, curAnot)) {
      int n = strlen(temp);
      if(n <= 3) continue;
      if(temp[0] == '#') continue;
      printf("%s", temp);
      vector<string> tokens;
      boost::algorithm::split(tokens, temp, boost::is_any_of(" ,()\"-':"), 
          boost::token_compress_on);
      for(int i = 0; i < (int) tokens.size(); ++i)
        TRACE(tokens[i]);

      continue;
      if(tokens[1] == "filename") {
        curNode = ImNode(tokens[2]);
        continue;
      }

      if(tokens[1] == "size") {
        curNode.lenX = cast<int>(tokens[7]);
        curNode.lenY = cast<int>(tokens[9]);
        curNode.channels = cast<int>(tokens[11]);
        continue;
      }

      if(tokens[0] == "Center" && tokens[1] == "point") {
        int cx = cast<int>(tokens[8]);
        int cy = cast<int>(tokens[9]);
        curNode.cX.push_back(cx);
        curNode.cY.push_back(cy);
        continue;
      }

      if(tokens[0] == "Bounding" && tokens[1] == "box") {
        int xmin = cast<int>(tokens[10]);
        curNode.Xmin.push_back(xmin);
        int ymin = cast<int>(tokens[11]);
        curNode.Ymin.push_back(ymin);
        int xmax = cast<int>(tokens[12]);
        curNode.Xmax.push_back(xmax);
        int ymax = cast<int>(tokens[13]);
        curNode.Ymax.push_back(ymax);
        continue;
      }

    }
    printf("windows: %d\n", (int) curNode.Ymax.size());
    imNodes.push_back(curNode);
  }


  /*
  vector<double> tex_features;
  for(int p = 0; p < 4; ++p) { // co-occ type 
    vector<TextBlock> t = getTextBlocks(argv[1], p);
    for(int i = 0; i < (int) t.size(); ++i) {
      vector<double> f = t[i].getFeatures();
      tex_features.insert(tex_features.end(), f.begin(), f.end());
    }
  }
  */
}

int main(int argc, char** argv) {
  getTrainingSet();

  /*
  Model model();

  printf("Dodano %d text znacajki\n", (int) tex_features.size());
  puts("Done\n");
  */

  return 0;
}
