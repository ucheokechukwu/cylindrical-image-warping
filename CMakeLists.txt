# used to compile and build warper.cpp 

cmake_minimum_required(VERSION 2.8)
project( warper )
find_package( OpenCV REQUIRED )
add_executable( warper warper.cpp )
target_link_libraries( warper ${OpenCV_LIBS} )
