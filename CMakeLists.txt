cmake_minimum_required(VERSION 4.0)

project(cbnn-playground)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_EXE_LINKER_FLAGS "-static")
endif ()


# External libraries
add_subdirectory(external)
include_directories(${FETCHCONTENT_BASE_DIR}/effolkronium_random-src/include)

# Internal libraries
add_subdirectory(lib)

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
        # src/ML/Neuron.cpp
        # src/ML/Layer.cpp
        # src/ML/MLP.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE raylib fmt imgui implot rlimgui ::tinyfiledialogs stb libml libmath libverlet)
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}
)

add_executable(prototype
        src/main.old.cpp
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



target_link_libraries(prototype PRIVATE raylib fmt imgui implot rlimgui ::tinyfiledialogs stb libml libmath libverlet)
set_target_properties(prototype PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}
)
