cmake_minimum_required(VERSION 2.8)
project(PR_PROJECT)
find_package(OpenCV REQUIRED)
add_executable(pls.cpp model.cpp Storage.cpp encoding.cpp maths.cpp encoding.cpp base64.cpp plsanje.cpp)
