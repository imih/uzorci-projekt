#pragma once

#include <vector>

#include <opencv2/ml/ml.hpp>

#include "../texture/TextBlock.h"
#include "../hogcolor/HOGBlock.h"
#include "model.h"

using std::vector;
using texture::TextBlock;
using hog::HOGBlock;

// returns the what blocks to use for the first stage 
void plsPerBlock(vector<vector <TextBlock> >&, vector<vector <TextBlock> >&,
    vector<vector <HOGBlock> >&, vector<vector<HOGBlock> >&);

// returns the what blocks to use for the second stage and how many features overall
void plsFull(vector<vector <TextBlock> >&, vector<vector <TextBlock> >&,
    vector<vector <HOGBlock> >&, vector<vector<HOGBlock> >&, 
    vector<vector <TextBlock> >&, vector<vector<HOGBlock> >&);

vector<float> getFeats(vector<TextBlock>&, vector<HOGBlock>&, int, bool);
