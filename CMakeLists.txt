cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(networks_ml)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(${CMAKE_SOURCE_DIR}/signal_sharing)
add_subdirectory(${CMAKE_SOURCE_DIR}/exchange_economy)
