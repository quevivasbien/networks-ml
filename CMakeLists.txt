cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(networks_ml)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(networks_ml networks_ml.cpp)
target_link_libraries(networks_ml "${TORCH_LIBRARIES}")
set_property(TARGET networks_ml PROPERTY CXX_STANDARD 17)
