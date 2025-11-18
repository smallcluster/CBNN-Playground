cmake_minimum_required(VERSION 4.0)

project(libverlet VERSION 0.0.1 LANGUAGES CXX)

set(HEADERS
        include/libverlet/verlet.h
)

set(SRCS
        src/verlet.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRCS} ${HEADERS})

# Math lib
target_link_libraries(${PROJECT_NAME} PRIVATE libmath)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
