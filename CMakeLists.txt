cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(networks_ml)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(lib networks.h networks.cpp ml.h ml.cpp)
target_link_libraries(lib "${TORCH_LIBRARIES}")

add_executable(main main.cpp)
target_link_libraries(main lib)

add_executable(two_player_test two_player_test.cpp)
target_link_libraries(two_player_test lib)

set_property(TARGET lib PROPERTY CXX_STANDARD 17)
