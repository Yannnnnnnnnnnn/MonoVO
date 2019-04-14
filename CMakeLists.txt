cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(monoVO)

find_package(OpenCV REQUIRED)

add_executable(monoVO main.cpp)

target_link_libraries(monoVO  ${OpenCV_LIBS} )
