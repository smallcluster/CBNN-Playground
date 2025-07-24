cmake_minimum_required(VERSION 3.10)

include(FetchContent)

project(CoolChicVis)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(ImDrawIdx=unsigned\ int)

# raylib
find_package(raylib QUIET)
if (NOT raylib_FOUND)
    FetchContent_Declare(
            raylib
            GIT_REPOSITORY https://github.com/raysan5/raylib.git
            GIT_TAG 5.5
            GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(raylib)
endif ()

# fmt
find_package(fmt QUIET)
if (NOT fmt_FOUND)
    FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 11.2.0
            GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(fmt)
endif ()

# spdlog
find_package(spdlog QUIET)
if (NOT spdlog_FOUND)
    FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.15.3
            GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(spdlog)
endif ()

# effolkronium_random
FetchContent_Declare(
        effolkronium_random
        GIT_REPOSITORY https://github.com/ilqvya/random.git
        GIT_TAG master
        GIT_SHALLOW 1
)
FetchContent_MakeAvailable(effolkronium_random)
include_directories(${FETCHCONTENT_BASE_DIR}/effolkronium_random-src/include)

# imgui
find_package(imgui QUIET)
if (NOT imgui_FOUND)
    FetchContent_Declare(
            imgui
            GIT_REPOSITORY https://github.com/ocornut/imgui.git
            GIT_TAG v1.92.0
            GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(imgui)
endif ()

set(IMGUI_PATH  ${FETCHCONTENT_BASE_DIR}/imgui-src)
file(GLOB IMGUI_SOURCES ${IMGUI_PATH}/*.cpp)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC ${IMGUI_PATH})

# implot
find_package(implot QUIET)
if (NOT implot_FOUND)
    FetchContent_Declare(
            implot
            GIT_REPOSITORY https://github.com/epezent/implot.git
            GIT_TAG master
            GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(implot)
endif ()

set(IMPLOT_PATH  ${FETCHCONTENT_BASE_DIR}/implot-src)
file(GLOB IMPLOT_SOURCES ${IMPLOT_PATH}/*.cpp)
add_library(implot STATIC ${IMPLOT_SOURCES})
target_include_directories(implot PUBLIC ${IMPLOT_PATH})
target_link_libraries(implot PRIVATE imgui)

# rlimgui
find_package(rlimgui QUIET)
if (NOT rlimgui_FOUND)
    FetchContent_Declare(
            rlimgui
            GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
            GIT_TAG main
            GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(rlimgui)
endif ()

set(RLIMGUI_PATH  ${FETCHCONTENT_BASE_DIR}/rlimgui-src)
file(GLOB RLIMGUI_SOURCES ${RLIMGUI_PATH}/*.cpp)
add_library(rlimgui STATIC ${RLIMGUI_SOURCES})
target_include_directories(rlimgui PUBLIC ${RLIMGUI_PATH})
target_link_libraries(rlimgui PRIVATE raylib imgui)


add_executable(${PROJECT_NAME}
        src/main.cpp
        # --- Utils ---
        src/Utils/GuiUtils.cpp
        src/Utils/GuiUtils.h
        src/Utils/ColorsUtils.cpp
        src/Utils/ColorsUtils.h
        # --- ML ---
        src/ML/Neuron.cpp
        src/ML/Neuron.h
        src/ML/Layer.cpp
        src/ML/Layer.h
        src/ML/MLP.cpp
        src/ML/MLP.h
        src/Utils/MathUtils.cpp
        src/Utils/MathUtils.h
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "-static")
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE raylib fmt imgui implot rlimgui spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32> )
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}
)