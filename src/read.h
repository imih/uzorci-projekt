#pragma once

#include <vector>

#include "texture/TextBlock.h"
#include "hogcolor/HOGBlock.h"

using std::vector;
using texture::TextBlock;
using hog::HOGBlock;

void readTex(vector<vector<TextBlock> >& v, bool ret);
void readHog(vector<vector<HOGBlock> >& v, bool ret);

void writeTex(vector<vector<TextBlock> >& v, bool ret);
void writeHog(vector<vector<HOGBlock> >& v, bool ret);
