cmake_minimum_required(VERSION 2.8)
project(PR_PROJECT)
find_package(OpenCV REQUIRED)
add_executable(train train.cpp hogcolor/HOGAndColorFeatures.cpp texture/TextBlock.cpp 
  texture/TextFeatures.cpp)
target_link_libraries(train ${OpenCV_LIBS})
