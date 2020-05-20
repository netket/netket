cmake_minimum_required(VERSION 3.12)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(NetKet LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

include(CTest)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NETKET_BUILD_TESTING "Build unit tests." OFF)
option(NETKET_USE_OPENMP "Use OpenMP for multithreading" OFF)
option(NETKET_USE_SANITIZER "Build test suite with Clang sanitizer" OFF)
option(NETKET_USE_BLAS "Use system BLAS instead of Eigen's implementation" ON)
option(NETKET_USE_LAPACK "Use system LAPACK instead of Eigen's implementation" OFF)

option(NETKET_NATIVE "Use -march=native" OFF)


set(NETKET_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling modules")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "[NetKet] CMAKE_BUILD_TYPE not specified, setting it to "
                 "Release. Use `-DCMAKE_BUILD_TYPE=...` to overwrite.")
  set(CMAKE_BUILD_TYPE Release)
endif()


#
# Dependencies
#
# json
################################################################################
ExternalProject_Add(
    json_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/json/include"
    URL "https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip"
    URL_HASH MD5=0dc903888211db3a0f170304cd9f3a89
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(json_project SOURCE_DIR)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE ${SOURCE_DIR})



# optional-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/optional-lite/nonstd/optional.hpp")
    file(DOWNLOAD
        "https://github.com/martinmoene/optional-lite/releases/download/v3.2.0/optional.hpp"
        "External/optional-lite/nonstd/optional.hpp")
endif()
add_library(optional_lite INTERFACE)
target_include_directories(optional_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/optional-lite")


# span-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/span-lite/nonstd/span.hpp")
    file(DOWNLOAD
        "https://github.com/martinmoene/span-lite/releases/download/v0.5.0/span.hpp"
        "External/span-lite/nonstd/span.hpp")
endif()
add_library(span_lite INTERFACE)
target_include_directories(span_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/span-lite")

# any
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/any/any.hpp")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/thelink2012/any/master/any.hpp"
        "External/any/any.hpp")
endif()
add_library(linb_any INTERFACE)
target_include_directories(linb_any
    INTERFACE "${CMAKE_BINARY_DIR}/External/any")


# Eigen3
################################################################################
ExternalProject_Add(
    eigen_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/Eigen3"
    URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz"
    URL_HASH MD5=77a2c934eaf35943c43ee600a83b72df
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_project SOURCE_DIR)
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 SYSTEM INTERFACE ${SOURCE_DIR})


# pybind11
###############################################################################
ExternalProject_Add(
    pybind11_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/pybind11"
    URL "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz"
    URL_HASH MD5=e2120c5b0e3c20a93f2dfcdc55026ba8
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
if (NOT NETKET_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5)
endif()
find_package(PythonLibsNew ${NETKET_PYTHON_VERSION} REQUIRED)

add_library(pybind11 INTERFACE)
ExternalProject_Get_Property(pybind11_project SOURCE_DIR)
target_include_directories(pybind11 SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_include_directories(pybind11 SYSTEM INTERFACE ${PYTHON_INCLUDE_DIRS})

if(APPLE AND DEFINED ENV{CONDA_PREFIX})
      TARGET_LINK_LIBRARIES(pybind11 INTERFACE "-undefined dynamic_lookup")
      message(STATUS "Building in conda environment on MAC")
else()
      target_link_libraries(pybind11 INTERFACE ${PYTHON_LIBRARIES})
endif()

# Greatly reduces the code bloat
target_compile_options(pybind11 INTERFACE "-fvisibility=hidden")

# MPI
################################################################################
find_package(MPI REQUIRED)
if(MPI_C_VERSION_MAJOR LESS 3)
message( FATAL_ERROR "NetKet requires at least MPI 3." )
endif()

# OpenMP
################################################################################
if(NETKET_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(NOT TARGET OpenMP::OpenMP_CXX)
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()
endif()

# Catch2
################################################################################
if (BUILD_TESTING AND NETKET_BUILD_TESTING)
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/Catch2/catch2/catch.hpp")
        file(DOWNLOAD
            "https://github.com/catchorg/Catch2/releases/download/v2.4.0/catch.hpp"
            "External/Catch2/catch2/catch.hpp")
    endif()
    add_library(Catch2 INTERFACE)
    target_include_directories(Catch2
        INTERFACE "${CMAKE_BINARY_DIR}/External/Catch2/catch2")
endif()


#
# NetKet
#

add_library(netket_lib INTERFACE)
target_include_directories(netket_lib INTERFACE Sources)
target_include_directories(netket_lib SYSTEM INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(netket_lib
    INTERFACE
        ${MPI_CXX_LIBRARIES}
        Eigen3
        nlohmann_json
        optional_lite
        span_lite
        linb_any
        ${CMAKE_DL_LIBS}
)
if(NETKET_USE_OPENMP)
    target_link_libraries(netket_lib INTERFACE OpenMP::OpenMP_CXX)
endif()


set(NETKET_WARNING_FLAGS
    -Wall -Wextra -pedantic
    -Wshadow
)
target_compile_options(netket_lib INTERFACE ${NETKET_WARNING_FLAGS})

if(NETKET_NATIVE)
    target_compile_options(netket_lib INTERFACE "-march=native")
endif()

if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(netket_lib INTERFACE -fcolor-diagnostics)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(netket_lib INTERFACE -fdiagnostics-color=always)
    endif()
endif()

if(NETKET_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(netket_lib INTERFACE EIGEN_USE_BLAS=1)
    target_link_libraries(netket_lib INTERFACE ${BLAS_LIBRARIES})
endif()

if(NETKET_USE_LAPACK)
    find_package(LAPACK REQUIRED)
    find_package(LAPACKE)

    if(LAPACKE_FOUND)
        message("Found LAPACKE.")
        target_compile_definitions(netket_lib INTERFACE EIGEN_USE_LAPACK=1 EIGEN_USE_LAPACKE=1)
        target_link_libraries(netket_lib INTERFACE ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
    else()
        message(WARNING "Found LAPACK, but not LAPACKE.")
        target_compile_definitions(netket_lib INTERFACE EIGEN_USE_LAPACK=1)
        target_link_libraries(netket_lib INTERFACE ${LAPACK_LIBRARIES})
    endif()
endif()


set(NETKET_SOURCES
    Sources/pynetket.cc
    Sources/Graph/lattice.cc
    Sources/Graph/custom_graph.cc
    Sources/Graph/abstract_graph.cc
    Sources/Graph/py_graph.cc
    Sources/Utils/exceptions.cc
    Sources/Utils/py_utils.cc
    Sources/Utils/random_utils.cc)

add_library(netket MODULE ${NETKET_SOURCES})

# A workaround for missing __cpu_model bug in gcc-5 and Clangs earlier than 6
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    target_link_libraries(netket_lib INTERFACE gcc_s gcc)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    string(REGEX REPLACE "^([0-9]+)\..*$" "\\1" NETKET_CLANG_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
    string(REGEX REPLACE "^[0-9]+\.([0-9])+.*$" "\\1" NETKET_CLANG_MINOR "${CMAKE_CXX_COMPILER_VERSION}")
    find_program(LLVM_CONFIG_EXECUTABLE
        NAMES
            llvm-config-${CMAKE_CXX_COMPILER_VERSION}
            llvm-config-${NETKET_CLANG_MAJOR}.${NETKET_CLANG_MINOR}
            llvm-config-${NETKET_CLANG_MAJOR})

    if (NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR
            "llvm-config: not found. Clang ${CMAKE_CXX_COMPILER_VERSION} has a bug. "
            "A workaround for it involves linking against libclang_rt.builtins-x86_64. "
            "To find that library, we need llvm-config.")
    endif()
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
        OUTPUT_VARIABLE NETKET_LLVM_LIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_library(NETKET_CLANG_RT
        NAMES clang_rt.builtins-x86_64
        HINTS ${NETKET_LLVM_LIBDIR}/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/linux)
    if(NOT NETKET_CLANG_RT)
        message(FATAL_ERROR
            "clang_rt: not found. Compilation will fail because "
            "of a known bug in Clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    message(STATUS "Found clang_rt: ${NETKET_CLANG_RT}")
    target_link_libraries(netket_lib INTERFACE ${NETKET_CLANG_RT})
endif()

target_link_libraries(netket PUBLIC netket_lib pybind11)
set_target_properties(netket PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")
add_dependencies(netket json_project eigen_project pybind11_project)
set_target_properties(netket PROPERTIES OUTPUT_NAME "_C_netket")


# if(NETKET_SANITIZER)
#     message(STATUS "[NetKet] Building python library with address and UB sanitizers")
#     target_compile_options(netket_lib
#         INTERFACE
#             -g -fno-omit-frame-pointer
#             -fsanitize=address -fsanitize=undefined
#     )
#     target_link_libraries(netket_lib
#         INTERFACE
#             -fsanitize=address -fsanitize=undefined
#     )
# endif()

#
# Testing
#

if(BUILD_TESTING AND NETKET_BUILD_TESTING)
    add_library(netket_test INTERFACE)
    target_link_libraries(netket_test INTERFACE netket_lib Catch2)
    enable_testing()
    add_subdirectory(Test)
endif()
