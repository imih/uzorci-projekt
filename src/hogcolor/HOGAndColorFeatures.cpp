#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <set>
#include <vector>
#include <opencv2/opencv.hpp>

#include "HOGBlock.h"

namespace hog {

  using std::vector;

  using cv::Mat; 
  using cv::Rect;


  const int ANGLE_CNT = 2;
  const int ANGLE_BIN_CNT = 12;
  const double PI = 4.0 * atan(1);

  enum {ANGLE_0, ANGLE_90, ANGLE_45, ANGLE_135};
  Mat kernels[ANGLE_CNT];

  void init_kernels() {
    kernels[0] = (cv::Mat_< short > (1, 3) << -1, 0, 1);
    kernels[1] = (cv::Mat_< short > (3, 1) << -1, 0, 1);
    // kernels[2] = (Mat_< short > (3, 3) << -1, 0, 0, 0, 0, 0, 0, 0, 1) * 0.5;
    // kernels[3] = (Mat_< short > (3, 3) << 0, 0, -1, 0, 0, 0, 1, 0, 0) * 0.5;

    //for (int i = 0; i < ANGLE_CNT; ++i)
    //  std::cout << kernels[i] << std::endl << std::endl;
  }

  void calc_gradients(Mat &image, Mat *grad) {
    init_kernels(); // prebacit u neki glavni init?

    int sz[] = {image.rows, image.cols};

    for (int i = 0; i < ANGLE_CNT; ++i) {
      grad[i].create(image.rows, image.cols, CV_16SC3);
      filter2D(image, grad[i], -1, kernels[i]);
    }
  }

  void calc_hog_and_color(Mat *grad, Vector<float> &features) {
    Mat planes[ANGLE_CNT][3];

    for (int i = 0; i < ANGLE_CNT; ++i)
      split(grad[i], planes[i]);

    vector< float > color_hist(3, 0.0);
    vector<vector< float > > angle_hist(3, vector<float>(ANGLE_BIN_CNT, 0.0));

    int r = grad[0].rows;
    int c = grad[0].cols;

    features = Vector<float>(3 * (ANGLE_BIN_CNT + 1));
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        vector< float > square_norms(3);
        for (int ch = 0; ch < 3; ++ch) {
          float dx = planes[ANGLE_0][ch].at< unsigned char >(i, j);
          float dy = planes[ANGLE_90][ch].at< unsigned char >(i, j);
          square_norms[ch] = dx * dx + dy * dy;

          float alpha = atan2(dy, dx);
          if (alpha < 0.0) alpha += PI;
          if (alpha >= PI) alpha = 0.0;

          alpha *= 180.0 / PI;

          int a = floor(alpha) - 1e-6;
          int A = a / (180 / ANGLE_BIN_CNT);

          assert(A >= 0 && A < ANGLE_BIN_CNT);
          angle_hist[ch][A] += sqrt(dx * dx + dy * dy); // dodat tezinski na susjede?
        }
        int best = max_element(square_norms.begin(), square_norms.end())
          - square_norms.begin();

        color_hist[best] += 1;
      }
    }

    int f_pos = 0;
    for(int ch = 0; ch < 3; ++ch) {
      for(int i = 0; i < ANGLE_BIN_CNT; ++i) {
        features.SetElement(f_pos, angle_hist[ch][i]);
        f_pos++;
      }
    }
    for (int i = 0; i < 3; ++i) {
      features.SetElement(f_pos, color_hist[i]);
      f_pos++;
    }
  }

  // izracunaj feature za sve blokove...
  // jel ok ak su svi kvadratni?
  // 12x12 do 64x64?

  const int BLOCK_SIZES_CNT = 8;

  const int block_sizes[BLOCK_SIZES_CNT][2] = {
    {12, 12},
    {16, 16},
    {20, 20},
    {26, 26},
    {32, 32},
    {40, 40},
    {48, 48},
    {64, 64}
  };

  int get_shift_len(int n) {
    int bit = 0;
    while ((1 << (bit + 1)) <= n) ++bit;
    return 1 << (bit - 1);
  }

  void calc_features(Mat &image, vector< HOGBlock > &feature_vectors, 
      std::set<int> skipped) {
    Mat grad[ANGLE_CNT];
    calc_gradients(image, grad);

    int r = image.rows;
    int c = image.cols;
    int block_id = -1;

    for (int bl = 0; bl < BLOCK_SIZES_CNT; ++bl) {
      int shift = get_shift_len(std::__gcd(block_sizes[bl][0], block_sizes[bl][1]));

      for (int i = 0; i + block_sizes[bl][1] <= r; i += shift) {
        for (int j = 0; j + block_sizes[bl][0] <= c; j += shift) {
          block_id++;
          if(skipped.count(block_id)) 
            continue;
          Mat blk_grad[ANGLE_CNT];

          for (int a = 0; a < ANGLE_CNT; ++a)
            blk_grad[a] = Mat(image, Rect(j, i, block_sizes[bl][0], block_sizes[bl][1]));

          Vector<float> feature_vector; 
          calc_hog_and_color(blk_grad, feature_vector);
          feature_vectors.push_back(HOGBlock(feature_vector, block_id++));
        }
      }
    }
  }

}

/*
   namespace {
// main
int main(int argc, char **argv) {
Mat image;
image = imread(argv[1], 1);

if (!image.data) {
printf("a di je slika?\n");
return -1;
}

Mat grad[ANGLE_CNT];
hog::calc_gradients(image, grad);
char buf[100];

vector< Mat > feature_vectors;
hog::calc_features(image, feature_vectors);
printf("imam %d feature vectora\n", (int)feature_vectors.size());

namedWindow(buf, CV_WINDOW_AUTOSIZE);
for (int i = 0; i < ANGLE_CNT; ++i) {
sprintf(buf, "grad %d", i);
imshow("gradijent", grad[i]);
waitKey(0);
}

return 0;
}
}
*/
