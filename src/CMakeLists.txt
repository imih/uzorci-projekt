cmake_minimum_required(VERSION 2.8)
project(PR_PROJECT)
find_package(OpenCV REQUIRED)
add_executable(train train.cpp hogcolor/HOGAndColorFeatures.cpp texture/TextBlock.cpp 
  texture/TextFeatures.cpp pls/pls.cpp pls/model.cpp pls/Storage.cpp 
  pls/maths.cpp pls/plsanje.cpp pls/base64.cpp pls/encoding.cpp read.cpp)
set (CMAKE_CXX_FLAGS "-std=c++11 -g -fsanitize=address")
set (CMAKE_CXX_LINK_FLAGS "-fsanitize=address")
target_link_libraries(train ${OpenCV_LIBS})
