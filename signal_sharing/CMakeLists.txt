find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(signals_lib networks.h networks.cpp ml.h ml.cpp)
target_link_libraries(signals_lib "${TORCH_LIBRARIES}")

add_executable(signals main.cpp)
target_link_libraries(signals signals_lib)

add_executable(two_player_test two_player_test.cpp)
target_link_libraries(two_player_test signals_lib)

add_executable(uncorrelated_bayesian_test uncorrelated_bayesian_test.cpp)
target_link_libraries(uncorrelated_bayesian_test signals_lib)

set_property(TARGET signals_lib PROPERTY CXX_STANDARD 17)
set_property(TARGET signals PROPERTY CXX_STANDARD 17)
