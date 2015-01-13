#pragma once

#include <set>
#include <vector>

#include <opencv2/ml/ml.hpp>

#include "../texture/TextBlock.h"
#include "../hogcolor/HOGBlock.h"

using std::vector;
using std::set;
using texture::TextBlock;
using hog::HOGBlock;

void plsPerBlock(vector<vector <TextBlock> >&, vector<vector <TextBlock> >&, set<int>&,
    vector<vector <HOGBlock> >&, vector<vector<HOGBlock> >&, set<int>&);

