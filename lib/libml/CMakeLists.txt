cmake_minimum_required(VERSION 3.5)

project(libml VERSION 0.0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} STATIC
        # Compute graphs
        src/compute/utils.cpp
        src/compute/nodes.cpp
        src/compute/graph.cpp
        # Generic
        src/losses.cpp
        src/dataset.cpp
        # Neural networks
        src/neural/mlp.cpp
        src/neural/neuron.cpp
        src/neural/optimizers.cpp
)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)