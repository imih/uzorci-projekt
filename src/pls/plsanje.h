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

// returns the what blocks to use for the first stage 
void plsPerBlock(vector<vector <TextBlock> >&, vector<vector <TextBlock> >&, set<int>&,
    vector<vector <HOGBlock> >&, vector<vector<HOGBlock> >&, set<int>&);

// returns the what blocks to use for the second stage and how many features overall
void plsFull(int, vector<vector <TextBlock> >&, vector<vector <TextBlock> >&, set<int>&,
    vector<vector <HOGBlock> >&, vector<vector<HOGBlock> >&, set<int>&);
