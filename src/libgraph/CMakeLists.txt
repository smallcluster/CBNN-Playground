cmake_minimum_required(VERSION 3.5)

project(libgraph VERSION 0.0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} src/graph.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_auto_type)
