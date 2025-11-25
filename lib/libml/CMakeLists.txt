cmake_minimum_required(VERSION 4.0)

project(libml VERSION 0.0.1 LANGUAGES CXX)

set(HEADERS
        # Compute graphs
        include/libml/compute/graph.h
        include/libml/compute/nodes.h
        # Neural networks
        include/libml/neural/activations.h
        include/libml/neural/aggregations.h
        include/libml/neural/dataset.h
        include/libml/neural/layers.h
        include/libml/neural/losses.h
        include/libml/neural/mlp.h
        include/libml/neural/neuron.h
        include/libml/neural/optimizers.h
)

set(SRCS
        # Compute graphs
        src/compute/nodes.cpp
        src/compute/graph.cpp
        # Neural networks
        src/neural/dataset.cpp
        src/neural/activations.cpp
        src/neural/aggregations.cpp
        src/neural/neuron.cpp
        src/neural/layers.cpp
        src/neural/mlp.cpp
        src/neural/optimizers.cpp
        src/neural/losses.cpp
)

# effolkronium_random
FetchContent_Declare(
        effolkronium_random
        GIT_REPOSITORY https://github.com/ilqvya/random.git
        GIT_TAG master
        GIT_SHALLOW 1
)
FetchContent_MakeAvailable(effolkronium_random)

add_library(${PROJECT_NAME} STATIC ${SRCS} ${HEADERS})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE effolkronium_random)
