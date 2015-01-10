#include <opencv2/opencv.hpp>
#include <vector>

namespace hog { 
  void calc_gradients(cv::Mat& , cv::Mat*);
  void calc_features(cv::Mat&, std::vector<cv::Mat>& );
};

