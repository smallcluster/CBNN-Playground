cmake_minimum_required(VERSION 3.10)

project(CoolChicVis)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(external)
include_directories(${FETCHCONTENT_BASE_DIR}/effolkronium_random-src/include)

add_executable(${PROJECT_NAME}
        src/main.cpp
        # --- Utils ---
        src/Utils/GuiUtils.cpp
        src/Utils/GuiUtils.h
        src/Utils/ColorsUtils.cpp
        src/Utils/ColorsUtils.h
        src/Utils/MathUtils.cpp
        src/Utils/MathUtils.h
        # --- ML ---
        src/ML/Neuron.cpp
        src/ML/Layer.cpp
        src/ML/MLP.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "-static")
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE raylib fmt imgui implot rlimgui spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32> )
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}
)