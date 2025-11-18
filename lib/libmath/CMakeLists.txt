cmake_minimum_required(VERSION 3.5)

project(libmath VERSION 0.0.1 LANGUAGES CXX)

set(HEADERS
        include/libmath/math.h
)

set(SRCS
        src/math.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRCS} ${HEADERS})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
