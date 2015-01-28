cmake_minimum_required(VERSION 2.8)
project(PR_PROJECT)
find_package(OpenCV REQUIRED)
add_executable(train train.cpp hogcolor/HOGAndColorFeatures.cpp texture/TextBlock.cpp 
  texture/TextFeatures.cpp pls/pls.cpp pls/model.cpp pls/Storage.cpp 
  pls/maths.cpp pls/plsanje.cpp pls/base64.cpp pls/encoding.cpp)
set (CMAKE_CXX_FLAGS "-std=c++11 -O2")
target_link_libraries(train ${OpenCV_LIBS})

add_executable(test test.cpp hogcolor/HOGAndColorFeatures.cpp texture/TextBlock.cpp 
  texture/TextFeatures.cpp pls/pls.cpp pls/model.cpp pls/Storage.cpp pls/maths.cpp
  pls/plsanje.cpp pls/base64.cpp pls/encoding.cpp)
target_link_libraries(test ${OpenCV_LIBS})

add_executable(classify classify.cpp hogcolor/HOGAndColorFeatures.cpp texture/TextBlock.cpp 
  texture/TextFeatures.cpp pls/pls.cpp pls/model.cpp pls/Storage.cpp pls/maths.cpp
  pls/plsanje.cpp pls/base64.cpp pls/encoding.cpp)
target_link_libraries(classify ${OpenCV_LIBS})
