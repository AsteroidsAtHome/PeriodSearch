cmake_minimum_required(VERSION 3.18)
project(asteroid_viewer CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- third-party (downloaded at configure time) --------------------------
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw  GIT_REPOSITORY https://github.com/glfw/glfw.git    GIT_TAG 3.4     GIT_SHALLOW TRUE)
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.5 GIT_SHALLOW TRUE)
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG v0.16 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)
endif()
FetchContent_GetProperties(implot)
if(NOT implot_POPULATED)
  FetchContent_Populate(implot)
endif()

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${implot_SOURCE_DIR}/implot.cpp ${implot_SOURCE_DIR}/implot_items.cpp)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${implot_SOURCE_DIR})
target_link_libraries(imgui_lib PUBLIC glfw)
target_compile_definitions(imgui_lib PRIVATE GLFW_INCLUDE_NONE)  # the GLFW backend needs no GL headers

# OpenGL functions are loaded at run time (glfwGetProcAddress); GLFW opens libGL itself on Linux,
# so only Windows links the GL library.
if(WIN32)
  find_package(OpenGL REQUIRED)
endif()

# ---- period_search numerics (compiled from the repository's period_search/*.c) ----
set(PERIOD_SEARCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../period_search" CACHE PATH "folder with period_search C sources")
if(NOT EXISTS "${PERIOD_SEARCH_DIR}/mrqmin.c")
  message(FATAL_ERROR "period_search sources not found in ${PERIOD_SEARCH_DIR} (set -DPERIOD_SEARCH_DIR=...)")
endif()
set(PS_C trifac areanorm sphfunc ellfit ludcmp lubksb mrqmin mrqcof curv blmatrix conv covsrt phasec matrix bright memory dot_product gauss_errc)
# The reference period_search allows only 1000 points per lightcurve (POINTS_MAX); the BOINC applications in use accept
# more (e.g. ATLAS lightcurves with 1500 points). The sources are copied to the build folder with a larger limit.
set(PS_POINTS_MAX 20000 CACHE STRING "max points in one lightcurve for the shape computation (period_search POINTS_MAX)")
set(PS_SRC "${CMAKE_CURRENT_BINARY_DIR}/ps_src")
foreach(f ${PS_C})
  if(f STREQUAL "mrqcof")
    # its work arrays (POINTS_MAX x MAX_N_PAR doubles) are local variables: with the larger limit they would not fit
    # on a thread's stack, so they become file-level statics (the shape computations run one at a time)
    file(READ "${PERIOD_SEARCH_DIR}/mrqcof.c" PS_MRQ)
    string(REGEX REPLACE "ytemp\\[POINTS_MAX\\+1\\],[ \t\r\n]*dytemp\\[POINTS_MAX\\+1\\]\\[MAX_N_PAR\\+1\\]," "" PS_MRQ2 "${PS_MRQ}")
    if(PS_MRQ2 STREQUAL PS_MRQ)
      message(FATAL_ERROR "mrqcof.c: work arrays not found, cannot raise POINTS_MAX")
    endif()
    string(REPLACE "double mrqcof(" "static double ytemp[POINTS_MAX+1], dytemp[POINTS_MAX+1][MAX_N_PAR+1];\n\ndouble mrqcof(" PS_MRQ2 "${PS_MRQ2}")
    file(CONFIGURE OUTPUT "${PS_SRC}/mrqcof.c" CONTENT "${PS_MRQ2}" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PERIOD_SEARCH_DIR}/mrqcof.c")
  else()
    configure_file("${PERIOD_SEARCH_DIR}/${f}.c" "${PS_SRC}/${f}.c" COPYONLY)
  endif()
endforeach()
file(GLOB PS_H "${PERIOD_SEARCH_DIR}/*.h")
foreach(h ${PS_H})
  get_filename_component(hn "${h}" NAME)
  if(NOT hn STREQUAL "constants.h")
    configure_file("${h}" "${PS_SRC}/${hn}" COPYONLY)
  endif()
endforeach()
file(READ "${PERIOD_SEARCH_DIR}/constants.h" PS_CONST)
string(REGEX REPLACE "#define[ \t]+POINTS_MAX[ \t]+[0-9]+" "#define POINTS_MAX ${PS_POINTS_MAX}" PS_CONST "${PS_CONST}")
file(CONFIGURE OUTPUT "${PS_SRC}/constants.h" CONTENT "${PS_CONST}" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PERIOD_SEARCH_DIR}/constants.h")
list(TRANSFORM PS_C PREPEND "${PS_SRC}/")
list(TRANSFORM PS_C APPEND ".c")
add_library(ps_numerics STATIC ${PS_C})
target_include_directories(ps_numerics PUBLIC ${PS_SRC})
target_compile_definitions(ps_numerics PUBLIC _USE_MATH_DEFINES)
if(MSVC)
  target_compile_options(ps_numerics PRIVATE /w)
else()
  target_compile_options(ps_numerics PRIVATE -w)
endif()
find_package(Threads REQUIRED)

# ---- the app ---------------------------------------------------------------
add_executable(asteroid_viewer
  src/main.cpp src/data.cpp src/model.cpp src/renderer.cpp src/gl_loader.cpp
  src/ps_core.cpp src/compute.cpp src/minkowski.cpp src/sbdb.cpp src/boinc.cpp src/cache.cpp src/orbitfit.cpp)
target_link_libraries(asteroid_viewer PRIVATE imgui_lib glfw ps_numerics Threads::Threads)
if(WIN32)
  target_link_libraries(asteroid_viewer PRIVATE OpenGL::GL winhttp)
endif()
target_compile_definitions(asteroid_viewer PRIVATE _USE_MATH_DEFINES)
if(MINGW)
  target_link_options(asteroid_viewer PRIVATE -static -static-libgcc -static-libstdc++ -mwindows)
elseif(WIN32)
  set_target_properties(asteroid_viewer PROPERTIES WIN32_EXECUTABLE ON)
  target_link_options(asteroid_viewer PRIVATE /ENTRY:mainCRTStartup)
endif()
