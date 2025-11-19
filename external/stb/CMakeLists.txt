cmake_minimum_required(VERSION 4.0)

project(stb VERSION 2.30 LANGUAGES CXX)

set(HEADERS
        include/stb/stb_image_write.h
)

set(SRCS
        src/stb_image.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRCS} ${HEADERS})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
