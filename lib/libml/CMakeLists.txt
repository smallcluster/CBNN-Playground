cmake_minimum_required(VERSION 3.5)

project(libml VERSION 0.0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} STATIC
        src/compute/utils.cpp
        src/compute/nodes.cpp
        src/compute/graph.cpp
        src/losses.cpp
        src/mlp.cpp
)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)