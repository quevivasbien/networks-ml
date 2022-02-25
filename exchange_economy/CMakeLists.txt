find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(exchange_lib exchange.h exchange.cpp ml_helper.h ml_helper.cpp)
target_link_libraries(exchange_lib "${TORCH_LIBRARIES}")

add_executable(exchange main.cpp)
target_link_libraries(exchange exchange_lib)

set_property(TARGET exchange_lib PROPERTY CXX_STANDARD 17)
