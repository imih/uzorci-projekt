cmake_minimum_required(VERSION 2.8)
project(PR_PROJECT)
find_package(OpenCV REQUIRED)
add_executable(train train.cpp hogcolor/HOGAndColorFeatures.cpp texture/TextBlock.cpp 
  texture/TextFeatures.cpp pls/pls.cpp pls/model.cpp pls/Storage.cpp 
  pls/maths.cpp pls/plsanje.cpp pls/base64.cpp pls/encoding.cpp read.cpp)
add_definitions(-std=c++11
-Wall -Wextra -pedantic -std=c++11 -O2 -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wlogical-op -Wcast-qual -Wcast-align -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -fstack-protector)
target_link_libraries(train ${OpenCV_LIBS})
