cmake_minimum_required(VERSION 3.5)

project(libverlet VERSION 0.0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} STATIC
        src/Verlet.cpp)

target_include_directories(${PROJECT_NAME}
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)