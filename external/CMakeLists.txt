cmake_minimum_required(VERSION 3.10)

include(FetchContent)

add_compile_definitions(ImDrawIdx=unsigned\ int)

# raylib
find_package(raylib QUIET)
if (NOT raylib_FOUND)
    FetchContent_Declare(
            raylib
            SOURCE_DIR raylib
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